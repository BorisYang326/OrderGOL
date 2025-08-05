import json
import logging
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# from src import utils
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image, ImageFont
from sklearn.cluster import KMeans
from torch import FloatTensor
import hashlib
import pickle
from src.configs import DATA_ROOT

logger = logging.getLogger(__name__)

KEY_MULT_DICT = {
    "x-y-w-h": {"x": 0, "y": 1, "w": 2, "h": 3},
    "xywh": {"x": 0, "y": 0, "w": 0, "h": 0},
}


def uniform_quantize_box(
    boxes: np.ndarray,
    width: float,
    height: float,
    size: int,
) -> np.ndarray:
    """Quantize the box coordinates to the grid.

    Args:
        size (int): quantization precision.
        boxes (np.ndarray): unquantized boxes given by annotations.
        width (float): image width given by annotations.
        height (float): image height given by annotations.

    Returns:
        np.ndarray: _description_
    """
    if not np.all((boxes >= 0) & (boxes <= 1)):
        boxes = normalize_bbox(boxes, width, height)
    # next take xywh to [0, size-1]
    boxes = (boxes * (size - 1)).round()

    return boxes.astype(np.int32)


def cluster_quantize_box(
    boxes: np.ndarray,
    kmeans_pos_centroids: dict[str, KMeans],
    N_bbox_cluster: int,
    shared_bbox_vocab: str,
) -> np.ndarray:
    """Quantize the box coordinates based on the K-Means clustering.

    Args:
        boxes (np.ndarray): unquantized boxes given by annotations.
        kmeans_pos_centroids (dict[str,KMeans]): dict of K-Means clustering centroids.
        shared_bbox_vocab (str): shared bbox vocab.

    Returns:
        np.ndarray: quantized boxes.
    """
    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 4)
    quantized_boxes = np.zeros_like(boxes)
    for idx, pos in enumerate(["x", "y", "w", "h"]):
        centroids = kmeans_pos_centroids[pos]
        dist = np.abs(boxes[:, idx] - centroids)
        cluster_idx = np.argmin(dist, axis=0)
        quantized_boxes[:, idx] = (
            cluster_idx + KEY_MULT_DICT[shared_bbox_vocab][pos] * N_bbox_cluster
        )
    return quantized_boxes.astype(np.int32)


def cluster_quantize_img(
    embedding: torch.Tensor, kmeans_centroids: np.ndarray, token_shift: int
) -> np.ndarray:
    """Discretize the embedding to token id.

    Args:
        embedding (torch.Tensor): embedding to be discretized (N*512).
        kmeans_centroids (np.ndarray): k-means centroids.
        token_shift (int): token shift for current vocab setting.

    Returns:
        int: token id (N*1).
    """
    distances = np.linalg.norm(
        kmeans_centroids[:, np.newaxis] - embedding.detach().cpu().numpy(),
        axis=2,
    )
    ann_img_tokens = np.argmin(distances, axis=0) + token_shift
    # ann_img_tokens = np.ones(distances.shape[1]) * 330
    assert isinstance(ann_img_tokens, np.ndarray)
    return ann_img_tokens


def load_img_centroids(img_cluster_N: int, patch_transform: str) -> np.ndarray:
    """Load image centroids from file.

    Args:
        img_cluster_N (int): image cluster number.
        patch_transform (str): patch transform mode.

    Returns:
        np.ndarray: image centroids.
    """
    kmeans_preprocess_mode = "pad" if patch_transform == "padding" else "rsp"
    img_n_cluster = img_cluster_N
    kmeans_img_postfix = (
        "/gt_train_kmeans/imgs/"
        + str(img_n_cluster)
        + "_"
        + kmeans_preprocess_mode
        + ".npy"
    )
    kmeans_img_centroids = np.load(DATA_ROOT + kmeans_img_postfix)

    logger.info(
        "Image centroids loaded from <{}>".format(DATA_ROOT + kmeans_img_postfix)
    )
    return kmeans_img_centroids


def load_bbox_centroids(bbox_cluster_N: int) -> dict[str, KMeans]:
    """Load bbox centroids from file.

    Args:
        bbox_cluster_N (int): bbox cluster number.

    Returns:
        dict[str, KMeans]: bbox centroids per position.
    """
    pos_n_cluster = bbox_cluster_N
    kmeans_pos_centroids = {}
    for idx, pos in enumerate(["x", "y", "w", "h"]):
        with open(
            DATA_ROOT
            + "/gt_train_kmeans/bbox/"
            + str(pos)
            + "_"
            + str(pos_n_cluster[idx])
            + ".pkl",
            "rb",
        ) as f:
            kmeans_pos_centroids[pos] = pickle.load(f)

    logger.info(
        "Position centroids loaded from <{}>".format(
            DATA_ROOT + "/gt_train_kmeans/bbox/*.pkl"
        )
    )
    return kmeans_pos_centroids

def gen_colors(num_colors: int) -> list:
    """Generate colors for visualization of different categories.

    <method from original repo of layout transformer>
    Args:
        num_colors (int): number of colors to generate.

    Returns:
        list: rgb triples list.
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)] for x in palette]
    return rgb_triples


def trim_tokens(tokens: np.ndarray, bos: int, eos: int, pad: int) -> np.ndarray:
    """Return layouts after trim padding token.

    Args:
        tokens (np.ndarray): layout with padding token, with shape (batch,max_len,hidden_dim)
        bos (int): bos token, usually 2^size+1
        eos (int): eos token, usually 2^size+2
        pad (int, optional): pad token, usually 2^size+3. Defaults to None.

    Returns:
        np.ndarray: layout after trim padding token, with shape (batch,max_len,hidden_dim)
    """
    bos_bool_tensor = tokens == bos
    bos_idx = np.where(bos_bool_tensor.cpu())[0]
    tokens = tokens[bos_idx[0] + 1 :] if len(bos_idx) > 0 else tokens
    eos_bool_tensor = tokens == eos
    eos_idx = np.where(eos_bool_tensor.cpu())[0]
    tokens = tokens[: eos_idx[0]] if len(eos_idx) > 0 else tokens
    tokens = tokens[tokens != pad]
    return tokens


def save_patches(
    crop_pil_list: Dict[str, Image.Image],
    save_dir: str,
    src_img: Image.Image,
    patch_cat: List[Tuple[int, str]],
    salient_scores: Optional[np.ndarray] = None,
    filter_ids: Optional[List[int]] = None,
    json_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Save patches and stack all patch tensors for later clustering.

    Args:
        crop_pil_list (List[Image.Image]): cropped patches from given image and bboxes.
        save_dir (str): save directory.
        src_img (str): source image name.
        patch_cat (Optional[List[str]], optional): patch category. Defaults to None.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    src_img_name = src_img.filename.split("/")[-1]
    pre_fix = src_img_name.split(".")[0]
    patches_info = {}
    patch_info_list = []
    for idx, (pil_img, cat) in enumerate(zip(crop_pil_list, patch_cat)):
        cat_id, cat_name = cat
        if filter_ids is not None:
            if cat_id not in filter_ids:
                continue
        patch_name = f"{pre_fix}_{idx}.png"
        sal_score = salient_scores[idx] if salient_scores is not None else -1.0
        patch_info = {
            "patch_name": patch_name,
            "image_name": src_img_name,
            "image_width": src_img.width,
            "image_height": src_img.height,
            "category": cat_name,
            "saliency_score": sal_score,
        }
        pil_img_path = os.path.join(save_dir, patch_name)
        pil_img.save(pil_img_path)
        patch_info_list.append(patch_info)
    patches_info[pre_fix] = patch_info_list
    if json_path is not None:
        existing_data = {}
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                existing_data = json.load(f)

        # 更新或添加 pre_fix 的信息
        existing_data.update(patches_info)

        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=4)


def normalize_bbox(boxes: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """Normalize the box coordinates to [0,1].

    Args:
        boxes (np.ndarray): annotation boxes with shape (n_boxes, 4),dtype=int.
        width (int): annotation canvas width.
        height (int): annotation canvas height.

    Returns:
        np.ndarray: normalized boxes with shape (n_boxes, 4),dtype=float.
    """
    # range of xy is [0, large_side-1]
    # range of wh is [1, large_side]
    # bring xywh to [0, 1]
    assert isinstance(src_w, int) and isinstance(
        src_h, int
    ), "width and height should be int."
    norm_boxes = deepcopy(boxes).astype(np.float32)
    if norm_boxes.ndim == 1:
        # if only one box
        norm_boxes = norm_boxes.reshape(-1, 4)
    norm_boxes[:, [2, 3]] = norm_boxes[:, [2, 3]] - 1
    norm_boxes[:, [0, 2]] = norm_boxes[:, [0, 2]] / float(src_w - 1)
    norm_boxes[:, [1, 3]] = norm_boxes[:, [1, 3]] / float(src_h - 1)
    norm_boxes = np.clip(norm_boxes, 0, 1)
    return norm_boxes


def denormalize_bbox(boxes: np.ndarray, tar_w: int, tar_h: int) -> np.ndarray:
    """Denormalize the box coordinates from [0,1] to [0, tar_w-1] or [0, tar_h-1].

    Args:
        boxes (np.ndarray): annotation boxes with shape (n_boxes, 4).
        tar_w (int): target canvas width.
        tar_h (int): target canvas height.

    Returns:
        np.ndarray: denormalized boxes with shape (n_boxes, 4).
    """
    boxes = boxes.copy()  #
    if boxes.ndim == 1:
        # if only one box
        boxes = boxes.reshape(-1, 4)
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (tar_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (tar_h - 1)
    boxes[:, [2, 3]] = boxes[:, [2, 3]] + 1
    ### clip and shift boxes if they are out of canvas
    # for box in boxes:
    #     x, y, w, h = box
    #     x = max(0, min(x, tar_w - w))
    #     y = max(0, min(y, tar_h - h))
    #     box[0] = x
    #     box[1] = y
    return boxes.astype(int)


def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor], is_center: bool = False):
    if is_center:
        xc, yc, w, h = bbox
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
    else:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def calculate_average_saliency(
    bboxes: np.ndarray, saliency_image_path: str
) -> np.ndarray:
    assert np.all((bboxes >= 0) & (bboxes <= 1)), "bboxes should be normalized."
    saliency_map = Image.open(saliency_image_path)
    saliency_array = np.array(saliency_map)
    mapped_bbox = denormalize_bbox(bboxes, saliency_map.width, saliency_map.height)
    saliency_values = []
    for bbox in mapped_bbox:
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        saliency_patch = saliency_array[y1:y2, x1:x2]
        average_saliency = np.mean(saliency_patch)
        saliency_values.append(average_saliency)
    return saliency_values


def trim_tokens(tokens, bos, eos, pad=None):
    bos_idx = np.where(tokens == bos)[0]
    tokens = tokens[bos_idx[0] + 1 :] if len(bos_idx) > 0 else tokens
    eos_idx = np.where(tokens == eos)[0]
    tokens = tokens[: eos_idx[0]] if len(eos_idx) > 0 else tokens
    # tokens = tokens[tokens != bos]
    # tokens = tokens[tokens != eos]
    if pad is not None:
        tokens = tokens[tokens != pad]
    return tokens


def resize_and_crop_image(image_path, target_width, target_height):
    with Image.open(image_path) as img:
        # opt_bbox = optimize_bbox(img, target_width / target_height)
        original_width, original_height = img.size
        ratio = max(target_width / original_width, target_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        crop_x1 = max(0, (img.size[0] - target_width) // 2)
        crop_y1 = max(0, (img.size[1] - target_height) // 2)
        crop_x2 = crop_x1 + target_width
        crop_y2 = crop_y1 + target_height
        img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        return img


def compute_avg_color_per_image(image: Image.Image):
    image = np.array(image)
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color.astype(int)


def compute_avg_color(image_files: List[str]):
    full_avg_color = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        avg_color = compute_avg_color_per_image(image)
        full_avg_color.append(avg_color)
    full_avg_color = np.array(full_avg_color)
    return tuple(full_avg_color.mean(axis=0).astype(int))


def compute_fill_text(
    font: ImageFont, bbox: Tuple[int, int, int, int], fill_str: str = "A"
) -> str:
    x1, y1, x2, y2 = bbox
    bbox_width, bbox_height = x2 - x1, y2 - y1
    char_x1, char_y1, char_x2, char_y2 = font.getbbox(fill_str)
    char_width, char_height = char_x2 - char_x1, char_y2 - char_y1
    num_chars_horizontal = bbox_width // char_width
    num_chars_vertical = bbox_height // char_height
    fill_text = (fill_str * num_chars_horizontal)[:num_chars_horizontal]
    return fill_text

def compute_hash(image):
    image_bytes = pickle.dumps(image)
    return hashlib.sha256(image_bytes).hexdigest()