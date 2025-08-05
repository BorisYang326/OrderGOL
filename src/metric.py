from collections import defaultdict
import multiprocessing
from functools import partial
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.distributions as tdist
from einops import rearrange, reduce, repeat
from prdc import compute_prdc
from pytorch_fid.fid_score import calculate_frechet_distance
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from torch import Tensor, FloatTensor
import datasets
from src.dataset.helpers.util import convert_xywh_to_ltrb
import logging


logger = logging.getLogger(__name__)
# from torch_geometric.utils import to_dense_adj


Feats = Union[FloatTensor, List[FloatTensor]]
Layout = Tuple[np.ndarray, np.ndarray]

# set True to disable parallel computing by multiprocessing (typically for debug)
# DISABLED = False
DISABLED = True


def __to_numpy_array(feats: Feats) -> np.ndarray:
    if isinstance(feats, list):
        # flatten list of batch-processed features
        if isinstance(feats[0], FloatTensor):
            feats = [x.detach().cpu().numpy() for x in feats]
        assert feats.ndim == 2, "feats should be 2D array."
        return np.concatenate(feats)
    else:
        feats = feats.detach().cpu().numpy()
        assert feats.ndim == 2, "feats should be 2D array."
        return feats


def compute_generative_model_scores(
    feats_real: Feats,
    feats_fake: Feats,
) -> Dict[str, float]:
    """
    Compute precision, recall, density, coverage, and FID.
    """
    feats_real = __to_numpy_array(feats_real)
    feats_fake = __to_numpy_array(feats_fake)

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)
    results = compute_prdc(real_features=feats_real, fake_features=feats_fake)
    results["fid"] = calculate_frechet_distance(
            mu_real, sigma_real, mu_fake, sigma_fake
        )
    results_ = {
        "fid": results["fid"],
        "coverage": results["coverage"],
        "density": results["density"],
        "precision": results["precision"],
        "recall": results["recall"],
    }
    # results_ = {'fid': results['fid'],'coverage': results['coverage']}
    return results_


def compute_saliency_aware_metrics(
    samples: dict,
    feature_label: datasets.ClassLabel,
) -> dict[str, list[float]]:
    """
    Compute saliency-aware metrics from samples dictionary.

    Args:
        samples: Dictionary containing sample information including:
            - visual.img: List of rendered images
            - visual.saliency: List of saliency maps
            - visual.layout: List of layout information in format [c, x, y, w, h]
        feature_label: ClassLabel object to convert between class IDs and names

    Returns:
        Dictionary with the following metrics as lists:
        - utilization: Utilization rate of space suitable for arranging elements,
          Higher values are generally better (in 0.0 - 1.0 range).
        - occlusion: Average saliency of areas covered by elements.
          Lower values are generally better (in 0.0 - 1.0 range).
        - readability: Non-flatness of regions that text elements are solely put on
          Lower values are generally better.
    """
    text_id = feature_label.str2int("textElement")
    results = defaultdict(list)

    for i, (img, saliency_map, layout_info) in enumerate(
        zip(
            samples["visual"]["img"],
            samples["visual"]["saliency"],
            samples["visual"]["layout"],
        )
    ):
        # Get image dimensions
        H, W = img.height, img.width

        # Convert saliency map to tensor if it's not already
        if not isinstance(saliency_map, torch.Tensor):
            saliency_map = torch.tensor(saliency_map)

        # Make sure saliency_map is 2D (H, W)
        if saliency_map.ndim > 2:
            saliency_map = saliency_map.squeeze()

        # Normalize saliency map to 0-1 range if needed
        if saliency_map.max() > 1.0:
            saliency_map = saliency_map / 255.0

        inv_saliency = 1.0 - saliency_map

        # Create empty mask for bounding boxes
        bbox_mask = torch.zeros((H, W), dtype=torch.bool)
        bbox_mask_text = torch.zeros((H, W), dtype=torch.bool)

        # For each element in the layout
        for element in layout_info:
            # Extract [c, x, y, w, h]
            c, cx, cy, w, h = element

            # Convert to absolute pixel coordinates
            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)

            # Clip to ensure within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            # Update bbox mask
            bbox_mask[y1:y2, x1:x2] = True

            # Update text mask if this is a text element
            if c == text_id:
                bbox_mask_text[y1:y2, x1:x2] = True

        # Calculate utilization - how well elements utilize low-saliency areas
        utilization_numerator = torch.sum(inv_saliency[bbox_mask])
        utilization_denominator = torch.sum(inv_saliency)
        if utilization_denominator > 0:
            utilization = (utilization_numerator / utilization_denominator).item()
            results["utilization"].append(utilization)
        else:
            results["utilization"].append(0.0)

        # Calculate occlusion - average saliency of covered areas
        occlusion_areas = saliency_map[bbox_mask]
        if len(occlusion_areas) > 0:
            occlusion = occlusion_areas.mean().item()
            results["occlusion"].append(occlusion)
        else:
            results["occlusion"].append(0.0)

        # Calculate readability - gradient in text-only areas
        # Convert image to numpy for OpenCV operations
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(img_gray, -1, 1, 0)
        grad_y = cv2.Sobel(img_gray, -1, 0, 1)
        grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5

        # Normalize gradient map
        if np.max(grad_xy) > 0:
            grad_xy = grad_xy / np.max(grad_xy)

        # Convert back to tensor
        grad_tensor = torch.from_numpy(grad_xy)

        # Calculate readability in text areas
        text_areas = grad_tensor[bbox_mask_text]
        if len(text_areas) > 0:
            readability = text_areas.mean().item()
            results["readability"].append(readability)
        else:
            results["readability"].append(0.0)

    return results


def _compute_wasserstein_distance_class(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
    n_categories: int = 25,
) -> float:
    categories_1 = np.concatenate([l[1] for l in layouts_1])
    counts = np.array(
        [np.count_nonzero(categories_1 == i) for i in range(n_categories)]
    )
    prob_1 = counts / np.sum(counts)

    categories_2 = np.concatenate([l[1] for l in layouts_2])
    counts = np.array(
        [np.count_nonzero(categories_2 == i) for i in range(n_categories)]
    )
    prob_2 = counts / np.sum(counts)
    return np.absolute(prob_1 - prob_2).sum()


def _compute_wasserstein_distance_bbox(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
) -> float:
    bboxes_1 = np.concatenate([l[0] for l in layouts_1]).T
    bboxes_2 = np.concatenate([l[0] for l in layouts_2]).T

    # simple 1-dimensional wasserstein for (cx, cy, w, h) independently
    N = 4
    ans = 0.0
    for i in range(N):
        ans += wasserstein_distance(bboxes_1[i], bboxes_2[i])
    ans /= N

    return ans


def compute_wasserstein_distance(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
    n_classes: int = 25,
) -> Dict[str, float]:
    w_class = _compute_wasserstein_distance_class(layouts_1, layouts_2, n_classes)
    w_bbox = _compute_wasserstein_distance_bbox(layouts_1, layouts_2)
    return {
        "wdist_class": w_class,
        "wdist_bbox": w_bbox,
    }


def compute_kendall_distance(score: torch.Tensor, gt: torch.Tensor) -> float:
    """Return kendall distance between score and ground truth.

    Args:
        score (torch.Tensor): predicted order with shape [n_elements]
        gt (torch.Tensor): ground truth order with shape [n_elements]

    Returns:
        float: kendall distance between score and ground truth.
    """
    # descending order from high to low.
    score_ind = torch.argsort(score, descending=True)
    n = len(score)
    if n != 1:
        # -1 as initial value maybe cause error.
        index_of = [-1] * n  # lookup into p2
        for i in range(n):
            v = int(gt[i].item())
            index_of[v] = i

        d = 0  # raw distance = number pair mis-orderings
        for i in range(n):  # scan thru score
            for j in range(i + 1, n):
                if index_of[score_ind[i]] > index_of[score_ind[j]]:  # replace "gt"
                    d += 1
        normer = n * (n - 1) / 2.0  # total num pairs
        nd = d / normer  # normalized distance
    else:
        nd = 0
    return nd


def compute_spearman_distance(score: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Return spearman distance between score and ground truth.

    Args:
        score (torch.Tensor): predicted order with shape [n_elements]
        gt (torch.Tensor): ground truth order with shape [n_elements]

    Returns:
        torch.Tensor: spearman distance (single tensor with dtype torch.float32)
        between score and ground truth.
    """
    # descending order from high to low.
    score_ranks = torch.argsort(-score)
    gt_ranks = torch.argsort(-gt)
    diff = score_ranks - gt_ranks
    square_sum = torch.sum(diff**2)
    if len(score) == 1:
        distance = torch.tensor(0.0, dtype=torch.float32)
    else:
        distance = square_sum / (len(score) - 1)
    return distance


def compute_spearman_correlation(score: np.ndarray, gt: np.ndarray) -> float:
    """Return spearman correlation between two order index.

    Args:
        score (torch.Tensor): predicted order with shape [n_elements]
        gt (torch.Tensor): ground truth order with shape [n_elements]

    Returns:
        float: spearman correlation rho between two order index.
    """
    n = score.size
    if n == 1:
        return 1.0
    score_tensor, gt_tensor = torch.from_numpy(score), torch.from_numpy(gt)
    distance = compute_spearman_distance(score_tensor, gt_tensor).item()
    rho = 1.0 - 6 * distance / (n * (n - 1))
    ### plot spearman correlation for orders per image, deprecated from dataset.py ###
    # plt.plot(sal_ras_diff)
    # plt.ylim(-1, 1)
    # plt.xlabel("image samples")
    # plt.ylabel("spearman_coff")
    # plt.title("order correlation")
    # plt.savefig("../figs/sal_ras_diff.png", dpi=300)
    return rho
