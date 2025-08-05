from math import e
from typing import Dict
import logging
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

SHIFT = 1e-5

logger = logging.getLogger(__name__)
# class NeuralSort(torch.nn.Module):
#     def __init__(self, tau=1.0, hard=False):
#         super(NeuralSort, self).__init__()
#         self.hard = hard
#         self.tau = tau

#     def forward(self, scores: Tensor):
#         """
#         scores: elements to be sorted. Typical shape: batch_size x n x 1
#         """
#         scores = scores.unsqueeze(-1)
#         bsize = scores.size()[0]
#         dim = scores.size()[1]
#         one = torch.cuda.FloatTensor(dim, 1).fill_(1)

#         A_scores = torch.abs(scores - scores.permute(0, 2, 1))
#         B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
#         scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(torch.cuda.FloatTensor)
#         C = torch.matmul(scores, scaling.unsqueeze(0))

#         P_max = (C - B).permute(0, 2, 1)
#         sm = torch.nn.Softmax(-1)
#         P_hat = sm(P_max / self.tau)

#         if self.hard:
#             P = torch.zeros_like(P_hat, device="cuda")
#             b_idx = (
#                 torch.arange(bsize)
#                 .repeat([1, dim])
#                 .view(dim, bsize)
#                 .transpose(dim0=1, dim1=0)
#                 .flatten()
#                 .type(torch.cuda.LongTensor)
#             )
#             r_idx = (
#                 torch.arange(dim)
#                 .repeat([bsize, 1])
#                 .flatten()
#                 .type(torch.cuda.LongTensor)
#             )
#             c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
#             brc_idx = torch.stack((b_idx, r_idx, c_idx))

#             P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
#             P_hat = (P - P_hat).detach() + P_hat
#         return P_hat


def _deterministic_neural_sort(s: torch.Tensor, tau: float) -> torch.Tensor:
    """The deterministic neural sort algorithm.

    Args:
        s (torch.Tensor): score tensor with size n * 1
        tau (float): temperature factor for Gumbel-Softmax.

    Returns:
        torch.Tensor: a permutation matrix without noise with size n * n.
    """
    device = s.device  # Detect the device type of the score 's'

    n = s.size()[1]
    one = torch.ones((n, 1), device=device)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
    scaling = n + 1 - 2 * (torch.arange(n, dtype=s.dtype, device=device) + 1)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)

    return P_hat


def _deterministic_soft_sort(s: torch.Tensor, tau: float) -> torch.Tensor:
    s_sorted = s.sort(descending=True, dim=1)[0]
    pairwise_distances = (s.transpose(1, 2) - s_sorted).abs().neg() / tau
    P_hat = pairwise_distances.softmax(-1)
    return P_hat


def deterministic_neural_sort(
    s: torch.Tensor, tau: float, version: str = "neuralsort"
) -> torch.Tensor:

    assert version in [
        "neuralsort",
        "softsort",
    ], "version should be either 'neuralsort' or 'softsort"
    if version == "neuralsort":
        return _deterministic_neural_sort(s, tau)
    else:
        return _deterministic_soft_sort(s, tau)


def stochastic_neural_sort(
    s: torch.Tensor, tau: float, version: str, scale_factor: float = 1.0
) -> torch.Tensor:
    """Return a permutation matrix P_hat.

    The core NeuralSort algorithm.
    P_hat that is differentiable w.r.t. s and approximates a permutation matrix P.


    Args:
        s (torch.Tensor): score tensor with size n * 1
        tau (float): temperature factor for Gumbel-Softmax.

    Return:
        P_hat (torch.Tensor): a permutation matrix with size n * n
    """

    def sample_gumbel(samples_shape, device, dtype=torch.float32, eps=1e-10):
        U = torch.rand(samples_shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + eps) + eps)

    # * modified for numeric stability
    #! eps = 1e-5
    batch_size, n, _ = s.size()
    #! log_s_perturb = torch.log(s+eps) + sample_gumbel([batch_size, n, 1], s.device, s.dtype)
    log_s_perturb_ = (
        torch.log(s)
        + sample_gumbel([batch_size, n, 1], s.device, s.dtype) * scale_factor
    )
    log_s_perturb = log_s_perturb_.view(batch_size, n, 1)
    P_hat = deterministic_neural_sort(log_s_perturb, tau, version)
    P_hat = P_hat.view(batch_size, n, n)

    return P_hat


def return_permutation(
    scores_: torch.Tensor,
    mask: torch.Tensor,
    epoch: int,
    is_train: bool,
    det_sort: bool = False,
    neural_sort_version: str = "neuralsort",
    noise_scale_factor: float = 1.0,
    score_norm_scale: float = 1.0,
    tau: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the permutation matrix and the relaxed permutation matrix.

    Args:
        scores_ (torch.Tensor): score for layouts per image with shape (B,S,1)
        mask (torch.Tensor): mask for padding with shape (B,S)
        epoch (int): current epoch.
        device (torch.device): device to use.
        det_sort (bool): whether use deterministic sort.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: permutation matrix and relaxed permutation matrix.
    """
    B, S, _ = scores_.shape
    # Start with identity matrices
    p_batch = torch.eye(S, device=scores_.device).unsqueeze(0).repeat(B, 1, 1)
    # time-dependent temperature
    if tau is None:
        tau = 1 / (1 + epoch**0.5)
    # scores shift to make sure all scores > 0
    stage_flag = "Train" if is_train else "Test"
    # ## Debugging
    sup_order_loss = {"raster": [], "saliency": []}
    # with torch.no_grad():
    #     _, buffer = scores_regulation(scores_, mask, score_norm_scale, reg_config)
    tie_counts = []
    s_inputs = []
    for b in range(B):
        tie_count = 0
        valid_len = mask[b].sum().item()
        # Extract valid scores
        scores = scores_
        s_valid = scores[b, :valid_len, :]
        if score_norm_scale>0:
            s_input = score_min_max_norm(s_valid, score_norm_scale)
        else:
            s_input = s_valid
        s_inputs.append(s_input)    
        if det_sort:
            # for deterministic sort, we dont need to shift scores since there's no log() operation.
            p_relaxed = deterministic_neural_sort(
                s_input.unsqueeze(0), tau, neural_sort_version
            )
        else:
            if score_norm_scale>0:
                s_input = scores_shift(s_input)
            else:
                s_input = s_input
            p_relaxed = stochastic_neural_sort(
                s_input.unsqueeze(0), tau, neural_sort_version, noise_scale_factor
            )  # Apply sorting to valid scores

        # Compute discrete permutation (p_discrete) from p_relaxed for valid scores
        p_discrete = torch.zeros_like(p_relaxed)
        indices = torch.argmax(p_relaxed, dim=-1)
        # assert unique_indices.numel() == indices.numel(), "Indices should be unique."
        if torch.unique(indices).numel() != indices.numel():
            tie_count += indices.numel() - torch.unique(indices).numel()
            indices = break_tie_argmax(p_relaxed)
            # if torch.unique(indices_).numel() != indices_.numel():
            #     raise ValueError("Indices should be unique.")
            # else:
            #     indices = indices_
        p_discrete.scatter_(-1, indices.unsqueeze(-1), 1)
        # Compute the final permutation matrix for valid scores
        p = p_relaxed + p_discrete.detach() - p_relaxed.detach()

        # Place the computed permutation matrix in the corresponding area of the identity matrix
        p_batch[b, :valid_len, :valid_len] = p.squeeze(0)

        ## Supervised order
        # ras_order = order_dics["raster"][b][mask[b]]
        # sal_order = order_dics["saliency"][b][mask[b]]
        # ras_loss_ = F.nll_loss(torch.log(p_relaxed.squeeze(0) + 1e-9), ras_order)
        # sal_order_ = F.nll_loss(torch.log(p_relaxed.squeeze(0) + 1e-9), sal_order)
        # sup_order_loss["raster"].append(ras_loss_)
        # sup_order_loss["saliency"].append(sal_order_)
        sup_order_loss["raster"].append(torch.tensor(0.0))
        sup_order_loss["saliency"].append(torch.tensor(0.0))
        tie_counts.append(tie_count)

        # ## Debugging
        # if b < 2 and epoch % 3 == 0:
        #     scores_array = s_input.squeeze(1).detach().cpu().numpy()
        #     np.set_printoptions(precision=5, suppress=True)
        #     scores_formatted = np.array2string(
        #         scores_array, formatter={"float_kind": lambda x: f"{x:.5f}"}
        #     )
        #     # sparse_loss_dic = buffer[b]
        #     # for key, value in sparse_loss_dic.items():
        #     #     sparse_loss_dic[key] = f"{value:.5f}"
        #     # logger.debug(
        #     #     f"Scores in <{stage_flag}> | [sample<{b}/{B}>,iters<{iters}>,epoch<{epoch}>]:\n{scores_formatted}.\n With sparse term:{sparse_loss_dic}."
        #     # )
        #     logger.debug(
        #         f"Scores in <{stage_flag}> | [sample<{b}/{B}>,iters<{iters}>,epoch<{epoch}>]:\n{scores_formatted}."
        #     )

    # sanity check
    # error_rate = calculate_error_rate(
    #     scores, p_batch, mask, is_train, score_norm_scale, iters, epoch
    # )
    sup_order_loss = {
        key: torch.stack(value).mean() for key, value in sup_order_loss.items()
    }
    error_rate = sum(tie_counts) / B
    return p_batch, error_rate, sup_order_loss, s_inputs
    # return p_relaxed, p


def scores_shift(scores: Tensor, shift: float = SHIFT):
    # to make sure scores all > 0
    scores_ = scores + torch.abs(torch.min(scores, dim=1, keepdim=True)[0]) + shift
    assert torch.all(torch.ge(scores_, 0)), "scores should be greater than 0"
    # return scores.view(scores.shape[0], -1, 1)
    return scores_


def score_min_max_norm(scores: Tensor, scale: float = 1.0):
    # Normalize scores using min-max normalization
    # Find the minimum and maximum scores for each sample
    assert scores.ndim == 2, "The score should be [S,1]."
    assert scores.size(1) == 1, "The score should be [S,1]."
    # Force scale to be a float
    scale = float(scale)
    n = scores.shape[0]
    min_scores = scores.min(dim=0, keepdim=True).values
    max_scores = scores.max(dim=0, keepdim=True).values
    # Handle the case where score has only one element or all elements are the same.
    if n == 1 or min_scores == max_scores:
        return torch.ones(scores.shape, device=scores.device) * scale
    # Normalize scores using min-max normalization
    scores = (scores - min_scores) / (max_scores - min_scores)
    # Check for nan values.
    assert not (torch.isnan(scores).any()), "The normalized score contains nan values."
    return scores * scale


def get_entropy(scores, mask):
    total_entropy = 0
    for i in range(scores.size(0)):
        valid_scores = scores[i][mask[i]]
        # Use log_softmax for numerical stability
        log_probabilities = F.log_softmax(valid_scores, dim=0)
        # Calculate entropy using log_probabilities
        entropy = -torch.exp(log_probabilities) * log_probabilities  # exp(log(p)) = p
        # Accumulate entropy for each sample
        total_entropy += entropy.sum()
    # Return average entropy by dividing total entropy by the number of samples
    return total_entropy / scores.size(0)


def sparse_loss(
    scores_, mask, score_norm_scale, entropy_weight=0.5, kl_weight=1.0, l2_weight=1.0
):
    batch_size = scores_.size(0)
    loss_per_sample = []
    buffer = []
    for i in range(batch_size):
        # Extract valid scores for the current sample
        s_valid = scores_[i][mask[i]]
        s_input = score_min_max_norm(s_valid, score_norm_scale).squeeze(-1)
        n_scores = s_input.size(0)
        # Skip if there are less than 2 valid scores
        if n_scores < 2:
            continue
        # s_pre_prob is the pre-softmax scores, to prevent softmax from overflow
        s_pre_prob = s_valid / torch.max(torch.abs(s_valid))
        s_prob = F.softmax(s_pre_prob, dim=0)
        s_log_prob = F.log_softmax(s_pre_prob, dim=0)
        s_uniform = torch.full_like(s_prob, fill_value=1 / n_scores)
        s_kl = F.kl_div(s_log_prob, s_uniform, reduction="sum")
        s_entropy = -torch.sum(s_prob * s_log_prob)
        # s_input is min-max normalized,used for calculating the L2 distance and margin.
        # Calculate the sum of squared differences between scores
        scores_diff = s_input.unsqueeze(0) - s_input.unsqueeze(1)
        scores_diff_sq = torch.square(scores_diff)

        # Use the upper triangular part of scores_diff_sq without the diagonal
        sum_scores_diff_sq = torch.triu(scores_diff_sq, diagonal=1).sum()

        # Normalize the loss by dividing it by the number of elements in the upper triangle minus the diagonal
        num_elements = n_scores * (n_scores - 1) / 2
        s_l2distance = -sum_scores_diff_sq / (num_elements * score_norm_scale**2)
        loss_ = l2_weight * s_l2distance + entropy_weight * s_entropy + kl_weight * s_kl
        loss_per_sample.append(loss_)
        buffer.append(
            {
                "loss": loss_.item(),
                "l2_distance": s_l2distance.item(),
                "entropy": s_entropy.item(),
                "kl-div": s_kl.item(),
            }
        )

    # Calculate the average loss within the batch
    if len(loss_per_sample) > 0:
        losses = torch.stack(loss_per_sample)
        assert torch.isnan(losses).sum() == 0, "Loss should not be nan."
        loss = losses.mean()
    else:
        loss = torch.tensor(0.0, device=scores_.device)
    # minimize the negative of the sum distance
    return loss, buffer


def margin_loss(scores_, mask, score_norm_scale, margin_ratio=1e-5):
    batch_size = scores_.size(0)
    loss_per_sample = []
    buffer = []
    for i in range(batch_size):
        # Extract valid scores for the current sample
        s_valid = scores_[i][mask[i]]
        s_input = score_min_max_norm(s_valid, score_norm_scale).squeeze(-1)
        n_scores = s_input.size(0)
        if n_scores < 2:
            continue
        # Calculate margins based on the maximum score
        # margin = score_norm_scale * margin_ratio / (n_scores - 1)
        margin = margin_ratio * score_norm_scale        
        # Calculate the absolute differences between scores
        scores_diff = torch.abs(s_input.unsqueeze(0) - s_input.unsqueeze(1))

        # Apply the margin mask
        margin_violations = F.relu(margin - scores_diff)

        # Use the upper triangular part without the diagonal
        upper_tri_violations = torch.triu(margin_violations, diagonal=1)

        # Sum up the violations and normalize
        loss_ = upper_tri_violations.sum() / (
            score_norm_scale * n_scores * (n_scores - 1) / 2
        )
        loss_per_sample.append(loss_)
        buffer.append({"margin": loss_.item()})

    # Calculate the average loss within the batch
    if len(loss_per_sample) > 0:
        losses = torch.stack(loss_per_sample)
        assert torch.isnan(losses).sum() == 0, "Loss should not be nan."
        loss = losses.mean()
    else:
        loss = torch.tensor(0.0, device=scores_.device)
    return loss, buffer


def scores_regulation(scores_, mask, scale, config: Dict):
    # alpha = config["alpha"]
    # beta = config["beta"]
    # gamma = config["gamma"]
    # zeta = config["zeta"]
    margin_ratio = config["margin_ratio"]
    # mg_loss, mg_buffer = margin_loss(scores_, mask, scale, margin_ratio)
    # sp_loss, sp_buffer = sparse_loss(scores_, mask, scale, beta, gamma, zeta)
    # for i, buf_dic in enumerate(sp_buffer):
    #     buf_dic["margin"] = mg_buffer[i]["margin"]
    # total_loss = mg_loss * alpha + sp_loss
    # # now sp_buffer contains all the information
    mg_loss, mg_buffer = margin_loss(scores_, mask, scale, margin_ratio)
    return mg_loss, mg_buffer


def calculate_error_rate(
    scores, p_batch, mask, is_train, score_norm_scale, iters, epoch, tolerance=1e-4
):
    B, S, _ = p_batch.shape
    per_sample_error_rate = []
    # Calculate the sum of rows and columns for each sample
    # rows_sum = p_batch.sum(dim=2)
    cols_sum = p_batch.sum(dim=1)
    stage_flag = "Train" if is_train else "Test"
    # Check if the absolute difference between the sum of rows/columns and 1 is greater than tolerance
    for b in range(B):
        # row_diffs = torch.abs(rows_sum[b] - 1) > tolerance
        col_diffs = torch.abs(cols_sum[b] - 1) > tolerance

        # Count the number of problematic rows and columns for each sample
        # problematic_rows = row_diffs.sum().item()
        problematic_cols = col_diffs.sum().item()

        # Record the total number of problematic rows and columns for each sample
        # problematic_samples_counts[b] = problematic_rows + problematic_cols
        valid_len = mask[b].sum().item()
        s_valid = scores[b, :valid_len, :]
        s_input = score_min_max_norm(s_valid, score_norm_scale)
        if problematic_cols > 0:
            # logger.debug(
            #     f"{stage_flag} | Sample {b}: {problematic_cols} problematic columns"
            # )
            scores_array = s_input.squeeze(1).detach().cpu().numpy()
            scores_formatted = np.array2string(
                scores_array, formatter={"float_kind": lambda x: f"{x:.5f}"}
            )
            logger.debug(
                f"Bad Scores in <{stage_flag}> with {problematic_cols} dup-columns.\n[sample<{b}/{B}>,iters<{iters}>,epoch<{epoch}>]:{scores_formatted}."
            )

        valid_total_cols = mask[b].sum().item()
        per_sample_error_rate.append(problematic_cols / valid_total_cols)

    # Weighted error rate is the total number of problematic rows/columns divided by the total possible number of rows/columns
    per_batch_error_rate = sum(per_sample_error_rate) / B
    return per_batch_error_rate


# DEPRECATED: This function is deprecated because it will cause consistent tie-breaking.
# def break_tie_argmax(p_relaxed_: torch.Tensor) -> torch.Tensor:
#     p_relaxed = p_relaxed_.squeeze(0)
#     N = p_relaxed.size(0)

#     # Initialize the tensor to hold the result indices
#     result_indices = torch.full((N,), -1, dtype=torch.long, device=p_relaxed.device)

#     # Process each row
#     for i in range(N):
#         # Get the row data
#         row = p_relaxed[i]
#         # Sort the row in descending order and get indices
#         sorted_indices = torch.argsort(row, descending=True)
#         # Allocate unique indices based on order
#         for idx in sorted_indices:
#             if idx not in result_indices:
#                 result_indices[i] = idx
#                 break
#     assert torch.unique(result_indices).numel() == N, "Indices should be unique."
#     return result_indices.unsqueeze(0)


def break_tie_argmax(
    p_relaxed_: torch.Tensor, max_iters: int = 10000, shift: float = 1e3
) -> torch.Tensor:
    p_relaxed = p_relaxed_.squeeze(0)
    N = p_relaxed.size(0)

    # Initialize a tensor to store the resulting indices
    result_indices = torch.full((N,), -1, dtype=torch.long, device=p_relaxed.device)
    indice_add_buffer = []
    # Process each row
    for i in range(N):
        # Get the row data
        # row = p_relaxed[i] * shift
        row = p_relaxed[i]
        # Sort the row data in descending order and get the indices
        sorted_indices = torch.argmax(row)
        current_max = row[sorted_indices]
        # all_possible_indices = torch.nonzero(row == current_max).squeeze(-1).tolist()
        all_possible_indices = (
            torch.nonzero(torch.isclose(row, current_max, atol=1e-10))
            .squeeze(-1)
            .tolist()
        )
        if len(all_possible_indices) == 1:
            result_indices[i] = sorted_indices
            indice_add_buffer.append(sorted_indices.item())
        else:
            cnt = 0
            all_possible_indices_ = all_possible_indices.copy()
            while True:
                random_idx = torch.randint(0, len(all_possible_indices_), (1,)).item()
                chosen_idx = all_possible_indices_[random_idx]
                if chosen_idx not in result_indices:
                    result_indices[i] = chosen_idx
                    indice_add_buffer.append(chosen_idx)
                    break
                else:
                    all_possible_indices_.remove(chosen_idx)
                    if len(all_possible_indices_) == 0:
                        # logger.warning(
                        #     f"Cannot find a unique index for row {i} after {cnt} iterations."
                        # )
                        torch.save(p_relaxed_, "p_relaxed_loop_empty.pt")
                        result_indices[i] = chosen_idx
                        indice_add_buffer.append(chosen_idx)
                        break
                cnt += 1
                if cnt > max_iters:
                    torch.save(p_relaxed_, "p_relaxed_loop.pt")
                    raise AssertionError(f"Exceeded maximum iterations {max_iters} while breaking ties.")        
    if torch.unique(result_indices).numel() != N:
        # Manually fix the result_indices unique issue
        missing_indices = torch.tensor(
            [i for i in range(N) if i not in result_indices], device=p_relaxed.device
        )
        # Get unique indices and counts
        uni_indices, counts = torch.unique(result_indices, return_counts=True)
        # Find duplicate indices
        duplicate_indices = uni_indices[counts > 1]
        # Find the positions of all duplicate indices
        relocated_pos = (result_indices[..., None] == duplicate_indices).nonzero(
            as_tuple=True
        )[0]
        # Merge duplicate indices and missing indices
        relocated_indices = torch.cat((duplicate_indices, missing_indices))
        # Randomly shuffle these positions
        relocated_pos = relocated_pos[torch.randperm(relocated_pos.size(0))]
        # Replace the duplicate positions in result_indices
        for pos, idx in zip(relocated_pos, relocated_indices):
            result_indices[pos] = idx
    try:
        assert torch.unique(result_indices).numel() == N, "Indices should be unique."
    except AssertionError as e:
        logger.warning(f"Indices should be unique.")
        torch.save(p_relaxed, "p_relaxed_uni.pt")
        # print(f"Indices should be unique.")
    # print(f"Indices: {result_indices}")
    return result_indices.unsqueeze(0)
