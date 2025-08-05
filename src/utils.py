import io
import cairosvg
import logging
import os
import pickle
import random
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from src.dataset import CrelloDataset
from src.scorer import Scorer
from tqdm import tqdm
import subprocess
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed

# from kmeans_pytorch import kmeans
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.optim import Optimizer, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler
from src.saliency.basnet import get_saliency_model, saliency_detect

logger = logging.getLogger(__name__)


class VoidScheduler(LRScheduler):
    """Scheduler that does not change anything."""

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)

    def step(self):
        pass


def linear_scheduler_lambda(max_epochs):
    return lambda x: 1 - x / max_epochs


def linear_scheduler_lambda_inverse(max_epochs, start_lr_ratio: float = 0.1):
    return lambda x: start_lr_ratio + (1 - start_lr_ratio) * x / (max_epochs - 1)


class LinearScheduler(object):
    """Linearly increase the learning rate from 0 to 1 in max_epochs."""

    def __init__(
        self, optimizer, max_epochs, inverse: bool = False, start_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        if inverse:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                linear_scheduler_lambda_inverse(max_epochs, start_lr_ratio),
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_scheduler_lambda(max_epochs)
            )

    def step(self):
        self.scheduler.step()


class CustomSchedulerWrapper:
    def __init__(self, optimizer, scheduler_cfg):
        self.optimizer = optimizer
        if scheduler_cfg._target_ == "torch.optim.lr_scheduler.SequentialLR":
            # Manually instantiate sub-schedulers
            sub_schedulers = []
            for sub_sched_cfg in scheduler_cfg.schedulers:
                sub_scheduler = instantiate(sub_sched_cfg)(optimizer=self.optimizer)
                sub_schedulers.append(sub_scheduler)

            # Create SequentialLR with instantiated sub-schedulers
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=self.optimizer,
                schedulers=sub_schedulers,
                milestones=scheduler_cfg.milestones,
            )
        else:
            # If not SequentialLR, instantiate normally
            self.scheduler = instantiate(scheduler_cfg)(optimizer=optimizer)

    def step(self, metrics=None):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metrics is None:
                raise ValueError("Metrics should be provided for ReduceLROnPlateau")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()


class SampleGenerator:
    def __init__(
        self,
        sample_func,
        batch_size: int = 1,
    ):
        # self.dataset = dataset
        self.sample_func = sample_func
        self.batch_size = batch_size

    def _generate(self):
        while True:
            yield self.sample_func(self.batch_size)


class EMA_Smoothing_Monitor:
    def __init__(self, beta: float = 0.0):
        self.beta = beta
        self.moving_average = None
        self.best_loss = float("inf")

    def check_update(self, current_loss: float) -> bool:
        if self.moving_average is None:
            # Initialize the moving average with the first observed loss value
            self.moving_average = current_loss
        else:
            # Update the moving average
            self.moving_average = (
                self.beta * self.moving_average + (1 - self.beta) * current_loss
            )

        # Check if there is improvement
        improve_flag = self.moving_average < self.best_loss

        if improve_flag:
            self.best_loss = self.moving_average

        return improve_flag


def set_seed(seed: int) -> None:
    """Set seed for numpy and torch.

    <method from original repo of layout transformer>
    Args:
        seed (int): random seed number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Return top k logits.

    <method from original repo of layout transformer>
    Args:
        logits (torch.tensor): _description_
        k (int): _description_

    Returns:
        torch.tensor: _description_
    """
    v, ix = torch.topk(logits, k)
    out = torch.clone(logits)
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def emb2indices(output: torch.Tensor, emb_layer: torch.nn.Embedding) -> torch.Tensor:
    """Return attributes index from embeddings.

    Args:
        output (torch.Tensor): with shape: [batch, sequence, emb_length]
        emb_layer (torch.nn.Embedding): embedding layer with shape: [vocab_size, emb_length]

    Returns:
        torch.Tensor: attributes index with shape: [batch, sequence]
    """
    batch, sequence = output.shape[:2]
    emb_weights = emb_layer.weight
    vocab_size, emb_length = emb_weights.shape
    out_indices_list = []
    # get indices from embeddings:
    for i in range(batch):
        out = output[i]
        out_index = torch.argmin(
            torch.abs(
                out.unsqueeze(1).expand([-1, vocab_size, -1])
                - emb_weights.unsqueeze(0).expand([sequence, -1, -1])
            ).sum(dim=2),
            dim=1,
        )
        out_indices_list.append(out_index)
    out_indices = torch.stack(out_indices_list)
    return out_indices


def ind_to_permutation_matrix(final_index: List[int]) -> torch.Tensor:
    """Return permutation matrix from a sorted indexes.

    Args:
        final_index (List[int]): corresponding list of index of a sorted list.

    Returns:
        torch.Tensor: corresponding permutation matrix with shape [n,n]
    """
    n = len(final_index)
    if n > 1:
        permutation_matrix = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            for j in range(n):
                if final_index[i] == j:
                    permutation_matrix[i, j] = 1.0
    else:
        permutation_matrix = torch.ones(1, dtype=torch.float32).unsqueeze(0)
    return permutation_matrix


def permutation_matrix_to_ind(permutation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert a permutation matrix to the corresponding index array.

    Args:
        permutation_matrix (torch.Tensor): The permutation matrix with shape [n, n].

    Returns:
        torch.Tensor: The corresponding list of indexes.
    """
    max_dim = 1
    if permutation_matrix.dim() == 3:
        # for B x n x n permutation matrix
        max_dim = 2
    # Find the column index where each row has its 1.0 value.
    _, indices = torch.max(permutation_matrix, dim=max_dim)
    assert isinstance(indices, torch.Tensor)
    return indices


def log_learning_rates(optimizer: Optimizer, logger: logging.Logger) -> None:
    """Log learning rates for each parameter group.

    Args:
        optimizer (torch.optim.Optimizer): current optimizer object.
        logger (logging.Logger): current logger object.
    """
    for param_group in optimizer.param_groups:
        logger.debug(f"Learning rate for {param_group['name']}: {param_group['lr']}")


def create_lr_lambda(epochs: int) -> Callable[[int], float]:
    """Return learning rate lambda function.

    Args:
        epochs (int): threshold epochs for learning rate decay.

    Returns:
        float: learning rate from lambda function.
    """
    return lambda epoch: 1.0 if epoch < epochs else np.exp(0.5 * (epochs - epoch))


def get_schedulers(
    scheduler_mode: str, optimizers: Tuple[Optimizer, Optimizer]
) -> Tuple[Optional[Any], Optional[Any]]:
    """Return schedulers for model and scorer.

    Args:
        scheduler_mode (str): scheduler mode, ['model_only', 'scorer_only', 'all']
        optimizers (Tuple[Optimizer, Optimizer]): model and scorer optimizers.

    Raises:
        ValueError: scheduler mode not supported.

    Returns:
        Tuple[lr_scheduler._LRScheduler, lr_scheduler._LRScheduler]: model and scorer schedulers.
    """
    model_optimizer, scorer_optimizer = optimizers
    model_scheduler = None
    scorer_scheduler = None
    if scheduler_mode == "model_only":
        model_scheduler = lr_scheduler.ExponentialLR(model_optimizer, gamma=0.5)
        logger.debug("Using model scheduler only.")
    elif scheduler_mode == "scorer_only":
        scorer_scheduler = lr_scheduler.LambdaLR(scorer_optimizer, create_lr_lambda(4))
        logger.debug("Using scorer scheduler only.")
    elif scheduler_mode == "all":
        model_scheduler = lr_scheduler.ExponentialLR(model_optimizer, gamma=0.5)
        scorer_scheduler = torch.optim.lr_scheduler.LambdaLR(
            scorer_optimizer, create_lr_lambda(9)
        )
        logger.debug("Using both schedulers.")
    else:
        logger.warning("No such scheduler mode.")
        raise ValueError("No such scheduler mode.")
    return model_scheduler, scorer_scheduler


def add_attribute_to_all_submodules(
    module: nn.Module, attribute_name: str, value: Any
) -> None:
    """Recursively adds an attribute to all submodules of a given module.

    Args:
        module (nn.Module): module that we want to add attribute.
        attribute_name (str): some attribute like boolean flag.
        value (Any): the default value for attribute setting.
    """
    setattr(module, attribute_name, value)
    for child_module in module.children():
        add_attribute_to_all_submodules(child_module, attribute_name, value)


def weight_update_back_hook(
    module_name: str,
    print_count: int,
) -> Callable[
    [
        torch.nn.Module,
        Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ],
    None,
]:
    """Return backward hook and logging function for given name module.

    Args:
        module_name (str): name of the module to hook.
        print_count (int): print gradient for every print_count times.

    Raises:
        ValueError: module has no romovable hook handle.

    Returns:
        torch.utils.hooks.RemovableHandle: hook handle.
    """

    def hook(
        module: torch.nn.Module,
        grad_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        grad_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> None:
        logger.debug(f"Current module: <{module}> from <{module_name}>.")

        if grad_input[0] is None:
            logger.warning("Gradient vanish detected in input grad!")
        if grad_output[0] is None:
            logger.warning("Gradient vanish detected in output grad!")
        elif torch.isnan(grad_input[0]).any():
            logger.warning("Gradient explosion detected in input grad!")
        else:
            assert isinstance(grad_input[0], torch.Tensor)
            in_grad_norm = torch.norm(grad_input[0])
            in_grad_min, in_grad_max = torch.min(grad_input[0]), torch.max(
                grad_input[0]
            )
            in_zero_ratio = (grad_input[0] == 0).float().mean()
            logger.debug(
                f"Received gradients info: <grad_norm: {in_grad_norm}>,"
                f"<grad_min: {in_grad_min}>,<grad_max: {in_grad_max}>,"
                f"<zero_ratio: {in_zero_ratio}> during backpropagation."
            )

        if grad_output[0] is None:
            logger.warning("Gradient vanish detected in output grad!")
        else:
            out_grad_norm = torch.norm(grad_output[0])
            out_grad_min, out_grad_max = torch.min(grad_output[0]), torch.max(
                grad_output[0]
            )
            out_zero_ratio = (grad_output[0] == 0).float().mean()
            logger.debug(
                f"Output gradients info to next module: <grad_norm: {out_grad_norm}>,"
                f"<grad_min: {out_grad_min}>,<grad_max: {out_grad_max}>,"
                f"<zero_ratio: {out_zero_ratio}> during backpropagation."
            )
        if hasattr(module, "_weight_update_back_hook_handle"):
            if isinstance(
                module._weight_update_back_hook_handle,
                torch.utils.hooks.RemovableHandle,
            ):
                module._weight_update_back_hook_handle.remove()
            else:
                # Handle the case where it's not a RemovableHandle.
                raise ValueError("Not a RemovableHandle!")

    return hook


def weight_update_forward_hook(
    module_name: str,
) -> Callable[
    [
        torch.nn.Module,
        Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ],
    None,
]:
    """Return forward hook and logging function for given name module.

    Args:
        module_name (str): name of the module to hook.

    Raises:
        ValueError: module has no romovable hook handle.

    Returns:
        torch.utils.hooks.RemovableHandle: hook handle.
    """

    def hook(
        module: torch.nn.Module,
        input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> None:
        logger.debug(f"********** Forward hook called from <{module_name}> **********")
        logger.debug("This module is called during forward pass!")
        logger.debug(f"Current input: {input}")
        logger.debug(f"Current input: {output}")
        if isinstance(
            module._weight_update_forward_hook_handle, torch.utils.hooks.RemovableHandle
        ):
            module._weight_update_forward_hook_handle.remove()
        else:
            # Handle the case where it's not a RemovableHandle.
            raise ValueError("Not a RemovableHandle!")

    return hook


def draw_computation_graph(
    writer: SummaryWriter,
    model: Any,
    inputs: torch.Tensor,
    model_name: str = "scorer",
    file_path_post: str = "../figs/forward_graph",
) -> None:
    """Draw computation graph for given model and inputs.

    Args:
        writer (SummaryWriter): writer to save the graph.
        model (Any): given model.
        inputs (torch.Tensor): given inputs.
        model_name (str, optional): current model name. Defaults to 'scorer'.
        file_path_post (str, optional): current saving path. Defaults to './figs/forward_graph'.
    """
    file_full_path = file_path_post + "_" + model_name
    writer.add_graph(model, inputs)
    writer.close()
    logger.info(
        f"Computation Graph for <{model_name}> has successfully saved in <{file_full_path}> "
    )


def gradient_check_histogram(
    writer: Any,
    model: Any,
    step: int,
    model_prefix: str,
    grad_record_interval: int = 100,
) -> None:
    """Check the gradient of the writer.

    Args:
        writer (Any): writer to save the graph (e.g., wandb).
        model (Any): the model to be checked.
        global_step (int): the global step of the writer.
        grad_record_interval (int): interval of steps to record gradients.
    """
    # Only log gradients every `grad_record_interval` steps
    if step % grad_record_interval == 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.log(
                    {
                        f"grads/{model_prefix}.{name}": wandb.Histogram(
                            param.grad.cpu().numpy()
                        ),
                        "steps/train": step,
                    }
                )
            else:
                if step == 10:
                    # just logging once
                    logger.debug(f"Gradient for {name} is None")


def logger_format_time(seconds: float) -> str:
    """Return formatted time string.

    Args:
        seconds (float): given seconds.

    Returns:
        str: formatted time string.
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.2f}s"


def draw_from_numpy(
    img_nd_arr: List[Image.Image],
    filename: str,
    save_path: str = "../figs/",
    bboxs: Optional[torch.Tensor] = None,
) -> None:
    """Draw a list of numpy arrays and save them as image files.

    Args:
        img_nd_arr (List[PIL.Image]): The list of PIL Image objects to be drawn.
        filename (str): The name of the file to be saved.
        save_path (str, optional): The path to save the file. Defaults to '../figs/'.
        bboxs (Optional[torch.Tensor], optional): The bounding boxes to be drawn. Defaults to None.
    """
    B = len(img_nd_arr)
    for i in range(B):
        img_nd_array = np.array(img_nd_arr[i])
        img_nd_array = (
            (img_nd_array - img_nd_array.min())
            / (img_nd_array.max() - img_nd_array.min())
            * 255
        )
        img_nd_array = img_nd_array.astype("uint8")

        dpi = 80
        height, width = img_nd_array.shape[:2]
        figsize = width / dpi, height / dpi

        # Create figure and plot tensor
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(img_nd_array)
        ax.axis("off")

        # Save figure as image file
        save_file = save_path + filename.split(".")[0] + "_" + str(i) + ".png"
        plt.savefig(save_file, bbox_inches="tight", pad_inches=0)

        # Close figure
        plt.close(fig)


def dequantize_box(
    bbox_normalized: torch.Tensor, width: int, height: int, precision: int
) -> torch.Tensor:
    """Dequantize the bounding bbox from normalized coordinates to absolute coordinates.

    Args:
        bbox_normalized (torch.Tensor): bounding bbox in normalized coordinates, shape (N, 4).
        width (int): raw image width.
        height (int): raw image height.
        precision (int): normalized image precision, and the normalized size = 2^precision.

    Returns:
        torch.Tensor: raw bounding bbox in absolute coordinates, shape (N, 4).
    """
    normal_size = pow(2, precision)
    bbox = bbox_normalized / (normal_size - 1)
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * (width - 1)
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * (height - 1)
    bbox[:, [2, 3]] = bbox[:, [2, 3]] + 1
    # NOTE: round operations will cause loss of information, but it is acceptable.
    bbox[:, 0] = torch.round(bbox[:, 0])
    bbox[:, 1] = torch.round(bbox[:, 1])
    bbox[:, 2] = torch.round(bbox[:, 2])
    bbox[:, 3] = torch.round(bbox[:, 3])

    assert isinstance(bbox, torch.Tensor)  # Add this to pass mypy check

    return bbox


def resolution_mapping(
    bbox: torch.Tensor,
    cur_img_size: Tuple[Any, Any],
    tar_img_size: Tuple[Any, Any],
) -> torch.Tensor:
    """Map the bounding boxes from current image size to raw image size.

    Args:
        bbox (torch.Tensor): bounding boxes, shape (N, 4).
        cur_img_size (Tuple[int|float, int|float]): current image size, (width, height).
        tar_img_size (Tuple[int|float, int|float]): target image size, (width, height).

    Returns:
        torch.Tensor: mapped bounding boxes, shape (N, 4).
    """
    cur_img_w, cur_img_h = cur_img_size
    tar_img_w, tar_img_h = tar_img_size
    assert isinstance(cur_img_w, int) or isinstance(
        cur_img_w, float
    ), "Current image size is not int or float"
    assert isinstance(tar_img_w, int) or isinstance(
        cur_img_w, float
    ), "Target image size is not int or float"
    width_scale = tar_img_w / cur_img_w
    height_scale = tar_img_h / cur_img_h

    # if width_scale != height_scale:
    #     logger.debug('*' * 80)
    #     logger.warning('Not proportionally mapping the bounding boxes!')
    #     logger.debug(f'width_scale: {width_scale}, height_scale: {height_scale}')
    #     logger.debug(f'cur_img_size: {cur_img_size}, tar_img_size: {tar_img_size}')

    assert isinstance(bbox, torch.Tensor), "Current bbox is not a torch tensor."
    bbox[:, 0] = bbox[:, 0] * width_scale  # x
    bbox[:, 2] = bbox[:, 2] * width_scale  # w
    bbox[:, 1] = bbox[:, 1] * height_scale  # y
    bbox[:, 3] = bbox[:, 3] * height_scale  # h
    # NOTE: round operations will cause loss of information, but it is acceptable.
    bbox[:, 0] = torch.round(bbox[:, 0])
    bbox[:, 1] = torch.round(bbox[:, 1])
    bbox[:, 2] = torch.round(bbox[:, 2])
    bbox[:, 3] = torch.round(bbox[:, 3])
    return bbox


def zero_one_normalization(score: torch.Tensor) -> torch.Tensor:
    """Normalize the score to [0,1] range.

    Args:
        score (torch.Tensor): The score to be normalized.

    Returns:
        torch.Tensor: The normalized score.
    """
    assert score.ndim == 1, "The score should be 1D."

    n = score.shape[0]

    # Handle the case where score has only one element or all elements are the same.
    if n == 1 or score.min() == score.max():
        return torch.ones(n, device=score.device)

    score = score - score.min()
    score = score / score.max()

    # Check for nan values.
    if torch.isnan(score).any():
        raise ValueError("The normalized score contains nan values.")

    return score


@torch.no_grad()
def order_analysis(
    scorer: Scorer,
    samples_gt: Dict[str, Dict[str, Tensor]],
    log_dir: str,
    device: torch.device,
):
    mask = samples_gt["elem"]["mask"][..., 0]
    seq = samples_gt["elem"]["seq"][..., 1:6]
    patches = samples_gt["elem"]["patch"]
    # Split the input into smaller batches
    id2order_dic = {}
    for i in tqdm(
        range(len(samples_gt["elem"]["seq"])),
        total=len(samples_gt["elem"]["seq"]),
        desc="Computing Scores..",
    ):
        seq_, mask_, patches_ = (
            seq[i].unsqueeze(0),
            mask[i].unsqueeze(0),
            patches[i].unsqueeze(0),
        )
        score_padded = scorer(seq_.to(device), mask_.to(device), patches_.to(device))
        actual_scores = rearrange(score_padded[mask_], "S 1 -> S")
        gt_id = samples_gt["visual"]["id"][i]
        score_ind = torch.argsort(actual_scores, descending=True).squeeze(-1)
        if score_ind.dim() == 0:
            assert (
                actual_scores.dim() == 1
            ), "actual_scores should be 1-dim for single element."
            score_ind = rearrange(score_ind, "-> 1")
        else:
            assert len(score_ind) == len(
                actual_scores
            ), "score_ind should be the same as argsort(scores)."
        id2order_dic.update({gt_id: score_ind.cpu().numpy()})
    os.makedirs(os.path.join(log_dir, "order"), exist_ok=True)
    with open(os.path.join(log_dir, "order/id2order.pkl"), "wb") as f:
        pickle.dump(id2order_dic, f)
    logger.info(
        f"Order has been saved successfully at : {os.path.join(log_dir, 'order/id2order.pkl')}"
    )


@torch.no_grad()
def generate_gen_samples(
    generator: SampleGenerator,
    # detokenize_func: Callable,
    # pad_token: int,
    dataset: CrelloDataset,
    sample_num: int = 1500,
    use_eos_mask: bool = True,
    N_workers: int = 6,
) -> Dict[str, Dict[str, Tensor]]:
    samples = {
        "seq": {"seq": [], "mask": []},
        "elem": {"seq": [], "mask": [], "patch": [], "canvas": []},
        "visual": {"id": [], "img": [], "svg": [], "layout": []},
    }
    sample_type = "GEN"
    decode_err_cnt = 0
    sample_iters = sample_num // generator.batch_size + 1

    def process_single_sample():
        nonlocal decode_err_cnt
        per_batch_err_cnt = 0
        while True:
            seq = next(generator._generate()).cpu()
            try:
                unpad_seq, canvas, mask_dic = dataset._detokenize(seq)
            except Exception as e:
                logger.error(f"Error: {e}")
                per_batch_err_cnt += 1
                continue

            if not torch.all(mask_dic["seq"]):
                break
            if per_batch_err_cnt >= 50:
                logger.warning("Too many decoding errors, break the loop.")
                break

        decode_err_cnt += per_batch_err_cnt
        pad_token_tensor = torch.tensor(dataset.pad_token, device=seq.device)
        if use_eos_mask:
            seq_ = seq.masked_fill(~mask_dic["seq"], pad_token_tensor)
        else:
            seq_ = seq
        seq_final, seq_mask_final = remove_bos_eos(
            seq_,
            mask_dic["seq"],
            dataset.bos_token,
            dataset.eos_token,
            dataset.pad_token,
            dataset.max_seq_length,
        )
        mask_dic["seq"] = seq_mask_final
        return seq_final, mask_dic, unpad_seq, canvas

    with ThreadPoolExecutor(max_workers=N_workers) as executor:
        future_to_sample = {
            executor.submit(process_single_sample): idx for idx in range(sample_iters)
        }

        with tqdm(
            total=sample_iters, desc=f"Generating per-batch <{sample_type}> samples.."
        ) as pbar:
            for future in as_completed(future_to_sample):
                seq_, mask_dic, unpad_seq, canvas = future.result()
                samples["seq"]["seq"].extend(seq_.squeeze(0))
                samples["seq"]["mask"].extend(mask_dic["seq"])
                samples["elem"]["seq"].extend(unpad_seq)
                samples["elem"]["mask"].extend(mask_dic["row"])
                samples["elem"]["canvas"].extend(canvas)
                pbar.update(1)

    samples["seq"]["seq"] = torch.stack(samples["seq"]["seq"])
    samples["seq"]["mask"] = torch.stack(samples["seq"]["mask"])
    samples["elem"]["seq"] = torch.stack(samples["elem"]["seq"])
    samples["elem"]["mask"] = torch.stack(samples["elem"]["mask"])
    samples["elem"]["canvas"] = torch.stack(samples["elem"]["canvas"])

    logger.info(
        f"Decode error rate: {decode_err_cnt}/{sample_iters} for <{sample_type}> stage."
    )

    return samples


def generate_gt_samples(
    dataloader, tokenize_func, detokenize_func, sample_num: int = 1500
):
    sample_type = "GT"
    sample_iters = sample_num // dataloader.batch_size + 1
    pbar = tqdm(
        enumerate(dataloader),
        total=sample_iters,
        desc=f"Generating per-batch <{sample_type}> samples..",
    )
    # TODO:RENAME the samples mapping.
    samples = {
        "seq": {"seq": [], "mask": []},
        "elem": {"seq": [], "mask": [], "patch": [], "canvas": []},
        "visual": {"id": [], "img": [], "svg": [], "layout": []},
    }
    for it, (seq, patches, mask, canvas_, order_dics) in pbar:
        if it >= sample_iters:
            break
        # seq, mask, patches, canvas_ = seq.to(device), mask.to(device), patches.to(device), canvas_.to(device)
        seq, mask = tokenize_func(seq, mask, canvas_["attr"], is_autoreg=False)
        # debug
        unpad_seq_tokenized, canvas_tokenized, mask_dic = detokenize_func(seq)
        # for i in range(dataloader.batch_size):
        #     seq_ = seq[...,:-1][i]
        #     mask_ = mask[...,1:][i]
        #     unpad_seq_tokenized,canvas_tokenized,mask_dic = detokenize_func(seq_)
        samples["seq"]["seq"].extend(seq)
        samples["seq"]["mask"].extend(mask)
        samples["elem"]["seq"].extend(unpad_seq_tokenized)
        samples["elem"]["mask"].extend(mask_dic["row"])
        # samples["elem"]["patch"].append(patches)
        # GT canvas: Dict[Dict[str,str],Dict[str,Tensor]]]
        samples["elem"]["canvas"].extend(canvas_tokenized)
        samples["visual"]["id"].extend(canvas_["id"])
    samples["seq"]["seq"] = torch.stack(samples["seq"]["seq"])
    samples["seq"]["mask"] = torch.stack(samples["seq"]["mask"])
    samples["elem"]["seq"] = torch.stack(samples["elem"]["seq"])
    samples["elem"]["mask"] = torch.stack(samples["elem"]["mask"])
    # samples["elem"]["patch"] = torch.stack(samples["elem"]["patch"])
    samples["elem"]["canvas"] = torch.stack(samples["elem"]["canvas"])
    return samples


def masked_avg_pooling(encoder_output, src_mask):
    # encoder_output has shape (B, S, H)
    # src_mask has shape (B, S), where True indicates the valid parts

    # Expand src_mask to match the dimensions of encoder_output
    src_mask = src_mask.unsqueeze(-1)  # becomes (B, S, 1)

    # Use src_mask to select the valid parts of encoder_output
    # Set elements at False positions to 0
    masked_output = encoder_output * src_mask

    # Calculate the number of valid tokens for each sample to avoid division by zero
    token_count = src_mask.sum(dim=1)  # (B, 1)
    token_count = token_count.where(
        token_count != 0, torch.ones_like(token_count)
    )  # prevent division by zero

    # Sum the valid parts and then divide by the number of valid tokens to get the average
    summed = masked_output.sum(dim=1)  # (B, H)
    averaged = summed / token_count  # (B, H)

    return averaged


def save_and_convert_svg(svg, id, log_dir, category, devNull):
    # Create directories for SVG and PNG files if they do not exist
    svg_dir = os.path.join(log_dir, f"svg/{category}")
    png_dir = os.path.join(log_dir, f"png/{category}")
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Define file paths for SVG and PNG
    svg_path = os.path.join(svg_dir, f"{category}_{id}.svg")
    png_path = os.path.join(png_dir, f"{category}_{id}.png")

    # Save the SVG file
    with open(svg_path, "w") as f:
        f.write(svg)

    # Convert SVG to PNG using Inkscape
    p = subprocess.Popen(
        ["inkscape", svg_path, "-e", png_path], stdout=devNull, stderr=devNull
    )
    p.wait()

    # Load the PNG image
    img = Image.open(png_path)

    # img.load()  # Ensure the image is loaded before closing
    # img.close()  # Close the image file to free up resources

    return svg_path, png_path, img


def remove_bos_eos(seqs, mask, bos_token, eos_token, pad_token, max_length):
    bos_eos_mask = (seqs != bos_token) & (seqs != eos_token)
    # lengths = bos_eos_mask.sum(dim=1)
    processed_seqs = torch.full(
        (seqs.size(0), max_length), pad_token, dtype=seqs.dtype, device=seqs.device
    )
    processed_masks = torch.full(
        (seqs.size(0), max_length), False, dtype=mask.dtype, device=mask.device
    )
    for i in range(seqs.size(0)):
        valid_elements = seqs[i][bos_eos_mask[i]]
        valid_mask = mask[i][bos_eos_mask[i]]
        length = min(len(valid_elements), max_length)
        processed_seqs[i, :length] = valid_elements[:length]
        processed_masks[i, :length] = valid_mask[:length]

    return processed_seqs, processed_masks


def calculate_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:
        return 0.0

    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )
    return total_norm


def load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"Failed to load cache from {path}.")
        return None
    except Exception as e:
        logger.warning(f"e")
        return None


def save_pickle(path: str, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def process_single_image(i, samples, dataset, render_partial=False):
    try:
        full_token = samples["elem"]["seq"][i]
        valid_token = full_token[samples["elem"]["mask"][i]].reshape(
            -1, dataset.N_var_per_element
        )
        svg, layouts = dataset.render(valid_token, samples["elem"]["canvas"][i])

        svg_bytes = svg.encode("utf-8")
        img_byte = cairosvg.svg2png(
            bytestring=svg_bytes, output_width=256, output_height=256
        )
        img = Image.open(io.BytesIO(img_byte))
        return svg, img, layouts
    except Exception as e:
        # return f"Error processing image {i}: {str(e)}", None, None
        logger.error(f"Error processing image {i}: {str(e)}")
        raise e


def gen_salmaps_to_samples(
    samples: Dict[str, Dict[str, Any]], device: torch.device, is_gt: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Add saliency maps to samples.

    Args:
        samples: Dictionary containing samples data
        device: The device to run the saliency model on
        is_gt: Whether these are ground truth samples (True) or generated samples (False)

    Returns:
        Updated samples dictionary with saliency maps added
    """

    sample_type = "GT" if is_gt else "GEN"
    logger.info(f"Predicting saliency maps for {sample_type} samples...")

    # Get the saliency model
    saliency_model = get_saliency_model().to(device)
    saliency_maps = []

    # Setup image conversion
    to_tensor = transforms.ToTensor()

    # Process each image
    for img in tqdm(
        samples["visual"]["img"], desc=f"Processing {sample_type} saliency maps"
    ):
        # Handle different image formats (RGBA vs RGB)
        if img.mode == "RGBA":
            # Convert RGBA to RGB by compositing over white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background

        # Convert to tensor and move to device
        img_tensor = to_tensor(img).to(device)

        # Process single image to get saliency map
        saliency_map = saliency_detect(saliency_model, img_tensor.unsqueeze(0))

        # Ensure the saliency map is float tensor (not bool tensor)
        saliency_map = saliency_map.float()

        saliency_maps.append(saliency_map.squeeze(0))

    # Add saliency maps to samples
    samples["visual"]["saliency"] = saliency_maps

    return samples
