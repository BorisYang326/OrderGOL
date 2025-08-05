import json
import os
from dataclasses import dataclass, field

import numpy as np
from torch import Tensor, LongTensor
from datasets import Dataset, Features
from typing import Optional, Tuple, Union

from .util import compute_hash
import random
import logging
from haishoku.haishoku import Haishoku

logger = logging.getLogger(__name__)


def handle_file_not_found_error(load_function):
    def wrapper(*args, **kwargs):
        try:
            return load_function(*args, **kwargs)
        except FileNotFoundError:
            logger.warning(
                f"File not found for {load_function.__name__} with args: {args}, kwargs: {kwargs}"
            )
            return None

    return wrapper


@dataclass
class ClusterStoreConfig:
    data_root: str
    cfg_type: str = field(default="json", init=True)
    all_patches_path: str = field(default="all_patches.json", init=False)
    is_canvas_cluster: bool = field(default=True)

    def __post_init__(self):
        self.all_patches_path = os.path.join(self.data_root, self.all_patches_path)
        self._json_types = ["img", "ff", "cc"]
        self._npy_types = ["fc", "fs", "cc", "x", "y", "w", "h"]
        if self.is_canvas_cluster:
            self._npy_types.extend(["cw", "ch"])
        for cluster_type in self._npy_types:
            setattr(
                self,
                f"{cluster_type}_cluster_centroids_path",
                os.path.join(
                    self.data_root,
                    f"weights/{cluster_type}_cluster/{cluster_type}_centers.npy",
                ),
            )
            setattr(
                self,
                f"{cluster_type}_cluster_labels_path",
                os.path.join(
                    self.data_root,
                    f"weights/{cluster_type}_cluster/{cluster_type}_labels.npy",
                ),
            )
        for cluster_type in self._json_types:
            setattr(
                self,
                f"{cluster_type}_cluster_result_path",
                os.path.join(
                    self.data_root,
                    f"weights/{cluster_type}_cluster/{cluster_type}_cluster_result.json",
                ),
            )


def get_cluster_store(config: ClusterStoreConfig, dataset: Optional[Dataset] = None):
    if config.cfg_type == "full":
        return FullClusterStore(config, dataset)
    elif config.cfg_type == "visual_only":
        return VisualOnlyClusterStore(config, dataset)
    else:
        raise NotImplementedError(f"Unknown cluster store type: {config.cfg_type}")


class FullClusterStore:
    def __init__(
        self, config: ClusterStoreConfig, dataset: Dataset, group_mapping: bool = False
    ):
        self._config = config
        self._dataset = dataset
        self._load_cluster_store_json()
        self._load_cluster_store_centroids()
        # self._available_fonts = self._load_font_family_ttflist()
        self._ff_freq = self._load_font_family_freq_list()
        logger.debug(f"Patches cluster grouping: <{group_mapping}>.")
        self._group_mapping = (
            self._load_json_data(
                os.path.join(self._config.data_root, "jsons/group.json")
            )
            if group_mapping
            else None
        )
        logger.info("Cluster store initialized.")

    def __call__(self, example: Features, patch_idx: int) -> dict:
        tokens = {}
        canvas_height = int(
            self._dataset.features["canvas_height"].int2str(example["canvas_height"])
        )
        is_text = example["font"][patch_idx] != 0
        tokens["ff"] = self._get_ff(example["font"][patch_idx]) if is_text else None
        tokens["fs"] = (
            self._get_fs(example["font_size"][patch_idx] / canvas_height)
            if is_text
            else None
        )
        tokens["fc"] = self._get_fc(example["color"][patch_idx]) if is_text else None
        tokens["img"] = (
            self._get_img(patch_idx, example["id"], example["type"][patch_idx])
            if not is_text
            else None
        )
        return tokens

    def _get_ff(self, rawlabel: int) -> int:
        ff_str = self._dataset.features["font"].feature.int2str(rawlabel)
        # for other fonts besides the top freq N_ff fonts, we use 0 to represent them
        try:
            return self._ff_data["ff2label"].get(ff_str, 0)
        except KeyError:
            logger.warning(f"KeyError: {ff_str} in ff_data")
            return 0

    def _get_fc(self, rgb: Tuple[int, int, int]) -> int:
        distance = np.sum((self._fc_centroids - rgb) ** 2, axis=1)
        return np.argmin(distance)

    def _get_cc(self, id: str) -> int:
        try:
            return int(self._cc_data[id])
        except KeyError:
            logger.warning(f"KeyError: {id} in cc_data")
            default_cc = np.array([255.0, 255.0, 255.0])
            return np.argmin(np.sum((self._cc_centroids - default_cc) ** 2, axis=1))

    def _get_canvas(self, example) -> np.ndarray:
        cc = self._get_cc(example["id"])
        return self._get_canvas_tokens(
            example["canvas_width"], example["canvas_height"], cc
        )

    def _get_canvas_tokens(
        self,
        cw_raw: int,
        ch_raw: int,
        cc_raw: Union[np.ndarray, Tuple[int, int, int], int],
    ) -> np.ndarray:
        if isinstance(cc_raw, int):
            cc = cc_raw
        else:
            if isinstance(cc_raw, tuple):
                cc_raw = np.array(cc_raw)
            cc = self.get_bc_tokens_from_RGB(cc_raw)
        if self._config.is_canvas_cluster:
            cw_dis = np.abs(self._cw_centroids - cw_raw)
            ch_dis = np.abs(self._ch_centroids - ch_raw)
            return np.array([np.argmin(cw_dis), np.argmin(ch_dis), cc])
        else:
            return np.array([cw_raw, ch_raw, cc])

    def _get_fs(self, fontsize: int) -> int:
        distance = np.abs(self._fs_centroids - fontsize)
        return np.argmin(distance)

    def _get_img(self, patch_id: int, canvas_id: str, type_id: str) -> int:
        type_ = self._dataset.features["type"].feature.int2str(type_id)
        # img_hash = compute_hash(pil_img)
        pid = f"{canvas_id}_{patch_id}"
        img_hash = self._img_pid2hash[type_].get(pid, None)
        try:
            return self._img_data[type_]["hash2cluster"][img_hash]
        except KeyError:
            logger.warning(f"KeyError: {type_}_pid:{pid}")
            return None

    def _load_cluster_store_json(self):
        for cluster_type in self._config._json_types:
            setattr(
                self,
                f"_{cluster_type}_data",
                self._load_json_data(
                    getattr(self._config, f"{cluster_type}_cluster_result_path")
                ),
            )
        self._img_pid2hash = self._load_json_data(
            os.path.join(
                self._config.data_root, "weights/img_cluster/image_pid2hash.json"
            )
        )

    def _load_cluster_store_centroids(self):
        for cluster_type in self._config._npy_types:
            setattr(
                self,
                f"_{cluster_type}_centroids",
                self._load_np_data(
                    getattr(self._config, f"{cluster_type}_cluster_centroids_path")
                ),
            )
            setattr(
                self,
                f"_{cluster_type}_labels",
                self._load_np_data(
                    getattr(self._config, f"{cluster_type}_cluster_labels_path")
                ),
            )

    @handle_file_not_found_error
    def _load_json_data(self, path: str) -> list:
        with open(path, "r") as f:
            return json.load(f)

    @handle_file_not_found_error
    def _load_np_data(self, path: str) -> np.ndarray:
        if path.endswith(".pkl"):
            path_ = path.replace(".pkl", ".npy")
            logging.info("Attempting to load numpy file instead of pickle file...")
            return np.load(path_, allow_pickle=True)
        return np.load(path, allow_pickle=True)

    def get_bbox_from_tokens(self, tokens: Tensor) -> np.ndarray:
        x_centers = self._bbox_cluster["x"][tokens[:, 0]]
        y_centers = self._bbox_cluster["y"][tokens[:, 1]]
        w_centers = self._bbox_cluster["w"][tokens[:, 2]]
        h_centers = self._bbox_cluster["h"][tokens[:, 3]]
        bbox = np.stack((x_centers, y_centers, w_centers, h_centers), axis=-1)
        if bbox.shape[0] != 1:
            bbox = bbox.reshape(-1, 4)
        return bbox

    def get_ff_ttf_from_token(self, token: int) -> Tuple[str, str]:
        key_is_str = isinstance(list(self._ff_data["label2ff"].keys())[0], str)
        if token == 0:
            # handle the case where the font is not ranked in the top N_ff fonts.
            font_names = list(self._ff_freq.keys())[len(self._ff_data["label2ff"]) :]
            probabilities = list(self._ff_freq.values())[
                len(self._ff_data["label2ff"]) :
            ]
            font_name = random.choices(font_names, weights=probabilities, k=1)[0]
        else:
            token_ = str(token) if key_is_str else token
            font_name = self._ff_data["label2ff"][token_]

        # # Regex-based matching
        # font_pattern = re.compile(re.escape(font_name.replace(' ', '').lower()))
        # matching_fonts = [f for f in self._available_fonts if font_pattern.search(f.replace(' ', '').lower())]

        # if matching_fonts:
        #     font_file = f"{matching_fonts[0]}.ttf"
        #     font_path = os.path.join(self._ttfs_dir, font_file)
        # else:
        #     font_file = f"{random.choice(self._available_fonts)}.ttf"
        #     font_path = os.path.join(self._ttfs_dir, font_file)
        #     logger.warning(f"Font name not found (even with regex matching): {font_name}. Using {font_file}")

        return font_name

    def get_fc_RGB_from_tokens(self, tokens: np.ndarray) -> np.ndarray:
        rgb_colors = self._fc_centroids[tokens]
        if rgb_colors.ndim == 1:
            rgb_colors = rgb_colors.reshape(-1, 3)
        noise = np.random.uniform(-10, 10, rgb_colors.shape)
        rgb_colors_with_noise = rgb_colors + noise
        rgb_colors_with_noise = np.clip(rgb_colors_with_noise, 0, 255)
        rgb_colors_rounded = np.rint(rgb_colors_with_noise).astype(int)
        color_tuples = [tuple(color) for color in rgb_colors_rounded]
        return color_tuples

    def get_bc_tokens_from_RGB(self, rgb: np.ndarray) -> int:
        distance = np.sum((self._cc_centroids - rgb) ** 2, axis=1)
        return np.argmin(distance)

    def get_bc_RGB_from_tokens(self, tokens: np.ndarray) -> np.ndarray:
        try:
            rgb_colors = self._cc_centroids[tokens]
            if rgb_colors.ndim == 1:
                rgb_colors = rgb_colors.reshape(-1, 3)
            # noise = np.random.uniform(-10, 10, rgb_colors.shape)
            # rgb_colors_with_noise = rgb_colors + noise
            # rgb_colors_with_noise = np.clip(rgb_colors_with_noise, 0, 255)
            rgb_colors_rounded = np.rint(rgb_colors).astype(int)
            color_tuples = [tuple(color) for color in rgb_colors_rounded]
        except IndexError:
            logger.warning(f"IndexError: {tokens}")
            color_tuples = [(255, 255, 255)]
        return color_tuples

    def get_fs_pt_from_tokens(
        self, tokens: np.ndarray, dpi_flag: bool = False
    ) -> np.ndarray:
        charbox_h = self._fs_centroids[tokens]
        # assert charbox_ratios < 1.0, f"charbox_ratios: {charbox_ratios}"
        if dpi_flag:
            return self._pixel2points(charbox_h).astype(int)
        else:
            return charbox_h

    def get_patch_from_token(
        self, token: Union[int, Tensor], cat: Union[int, Tensor], top_k: int = 20
    ) -> Optional[np.ndarray]:
        if isinstance(token, Tensor):
            token = token.item()
        if isinstance(cat, Tensor):
            cat = cat.item()
        image_prefix = os.path.join(self._config.data_root, "images/")
        cat_alias = "type" if "type" in self._dataset.features.keys() else "category"
        type_ = self._dataset.features[cat_alias].feature.int2str(cat)
        postfix_ = "png" if cat_alias == "type" else "jpg"
        if cat_alias == "type":
            group_id = type_
        else:
            group_id = (
                self._group_mapping["categories"][type_]
                if self._group_mapping["categories"].get(type_, None)
                else type_
            )
        try:
            cluster_fullimages = self._img_data[str(group_id)]["cluster2hash"][
                str(token)
            ]
        except KeyError:
            full_candidates = list(self._img_data[group_id]["hash2cluster"].keys())
            selected_image_ = random.choices(full_candidates)[0]
            selected_image = selected_image_.split(".")[0] + f".{postfix_}"
            return os.path.join(image_prefix, f"{group_id}/{selected_image}")
        if not cluster_fullimages:
            return None
        cluster_images = sorted(cluster_fullimages, key=lambda x: x["distance"])[:top_k]
        distances = np.array([img["distance"] for img in cluster_images])
        # probabilities = [1 / d for d in distances]
        # probabilities /= np.sum(probabilities)
        selected_image = random.choices(
            cluster_images, weights=self._softmax(distances), k=1
        )[0]
        return os.path.join(
            image_prefix, f"{group_id}/{selected_image['hash']}.{postfix_}"
        )

    def get_canvas_size_from_token(
        self, tokens: Union[np.ndarray, Tensor]
    ) -> Tuple[int, int]:
        if self._config.is_canvas_cluster:
            if isinstance(tokens, Tensor):
                tokens = tokens.cpu().numpy()
            cw_token, ch_token = tokens[1], tokens[2]
            cw_real, ch_real = int(self._cw_centroids[cw_token]), int(
                self._ch_centroids[ch_token]
            )
        else:
            cw_real = int(
                self._dataset.features["canvas_width"].int2str(tokens[1].item())
            )
            ch_real = int(
                self._dataset.features["canvas_height"].int2str(tokens[2].item())
            )
        return cw_real, ch_real

    def _pixel2points(self, pixels, dpi=72):
        points = (pixels * 72) / dpi
        return points

    def _softmax(self, distances, tau=0.1):
        adjusted_neg_exp_distances = np.exp(-distances / tau)
        return adjusted_neg_exp_distances / np.sum(adjusted_neg_exp_distances)

    @property
    def _bbox_cluster(self) -> list:
        return {key: getattr(self, f"_{key}_centroids") for key in ["x", "y", "w", "h"]}

    # @property
    # def _ttfs_dir(self):
    #     return os.path.join(self._config.data_root.split('filter')[0],'font/raw')

    # def _load_font_family_ttflist(self):
    #     available_fonts_ = os.listdir(self._ttfs_dir)
    #     available_fonts = [font_.split('.')[0] for font_ in available_fonts_]
    #     return available_fonts

    def _load_font_family_freq_list(self):
        ff_freq = self._load_json_data(
            os.path.join(self._config.data_root, "weights/ff_cluster/ff_freq.json")
        )
        return ff_freq


class VisualOnlyClusterStore:
    def __init__(
        self, config: ClusterStoreConfig, dataset: Dataset, group_mapping: bool = False
    ):
        self._config = config
        self._dataset = dataset
        self._dataset_name = self._config.data_root.split('/')[-1]
        assert self._dataset_name in ["cgl", "clay"], f"Dataset name: {self._dataset_name} is not supported."
        self._config._json_types = ["img"]
        self._config._npy_types = ["cc", "cw", "ch", "x", "y", "w", "h"]
        self._load_cluster_store_json()
        self._load_cluster_store_centroids()
        # self._available_fonts = self._load_font_family_ttflist()
        logger.debug(f"Patches cluster grouping: <{group_mapping}>.")
        self._group_mapping = (
            self._load_json_data(
                os.path.join(self._config.data_root, "jsons/group.json")
            )
            if group_mapping
            else None
        )
        logger.info("Cluster store initialized.")

    def __call__(self, example: Features, idx: int, patch_idx: Optional[int]=None) -> dict:
        tokens = {}
        if patch_idx is not None:
            tokens["img"] = (self._get_img(patch_idx, example["id"], example["type"][idx]))
        return tokens

    def _get_canvas(self, example) -> np.ndarray:
        cc = self._get_cc(example['id'])
        if self._dataset_name == "cgl":
            return np.array([0, 0, cc])
        else:   
            return self._get_canvas_tokens(
                example["canvas_width"], example["canvas_height"], int(cc)
        )

    def _get_cc(self, id: str) -> int:
        default_cc = np.array([255.0, 255.0, 255.0])
        middlefix_ = "preview" if self._dataset_name == "cgl" else "combined"
        preview_image_path = os.path.join(self._config.data_root, middlefix_, f"{id}.jpg")
        if not os.path.exists(preview_image_path):
            logger.warning(f"Preview image not found for {id}")
            cc = default_cc
        else:
            dominant_color = Haishoku.loadHaishoku(preview_image_path).dominant
            cc = np.array(dominant_color)
        return np.argmin(np.sum((self._cc_centroids - cc) ** 2, axis=1))

    def _get_canvas_tokens(
        self,
        cw_raw: int,
        ch_raw: int,
        cc_raw: Union[np.ndarray, Tuple[int, int, int], int],
    ) -> np.ndarray:
        if isinstance(cc_raw, int):
            cc = cc_raw
        else:
            if isinstance(cc_raw, tuple):
                cc_raw = np.array(cc_raw)
            cc = self.get_bc_tokens_from_RGB(cc_raw)
        if self._config.is_canvas_cluster:
            cw_dis = np.abs(self._cw_centroids - cw_raw)
            ch_dis = np.abs(self._ch_centroids - ch_raw)
            return np.array([np.argmin(cw_dis), np.argmin(ch_dis), cc])
        else:
            return np.array([cw_raw, ch_raw, cc])

    def _get_img(self, patch_id: int, canvas_id: str, type_id: str) -> int:
        type_ = self._dataset.features["type"].feature.int2str(type_id)
        pid = f"{canvas_id}-{patch_id}"
        try:
            return self._img_data[type_]["hash2cluster"][pid]
        except KeyError:
            logger.warning(f"KeyError: {type_}_pid:{pid}")
            return None

    def _load_cluster_store_json(self):
        for cluster_type in self._config._json_types:
            setattr(
                self,
                f"_{cluster_type}_data",
                self._load_json_data(
                    getattr(self._config, f"{cluster_type}_cluster_result_path")
                ),
            )

    def _load_cluster_store_centroids(self):
        for cluster_type in self._config._npy_types:
            setattr(
                self,
                f"_{cluster_type}_centroids",
                self._load_np_data(
                    getattr(self._config, f"{cluster_type}_cluster_centroids_path")
                ),
            )
            setattr(
                self,
                f"_{cluster_type}_labels",
                self._load_np_data(
                    getattr(self._config, f"{cluster_type}_cluster_labels_path")
                ),
            )

    @handle_file_not_found_error
    def _load_json_data(self, path: str) -> list:
        with open(path, "r") as f:
            return json.load(f)

    @handle_file_not_found_error
    def _load_np_data(self, path: str) -> np.ndarray:
        if path.endswith(".pkl"):
            path_ = path.replace(".pkl", ".npy")
            logging.info("Attempting to load numpy file instead of pickle file...")
            return np.load(path_, allow_pickle=True)
        return np.load(path, allow_pickle=True)

    def get_bbox_from_tokens(self, tokens: Tensor) -> np.ndarray:
        x_centers = self._bbox_cluster["x"][tokens[:, 0]]
        y_centers = self._bbox_cluster["y"][tokens[:, 1]]
        w_centers = self._bbox_cluster["w"][tokens[:, 2]]
        h_centers = self._bbox_cluster["h"][tokens[:, 3]]
        bbox = np.stack((x_centers, y_centers, w_centers, h_centers), axis=-1)
        if bbox.shape[0] != 1:
            bbox = bbox.reshape(-1, 4)
        return bbox

    def get_bc_tokens_from_RGB(self, rgb: np.ndarray) -> int:
        distance = np.sum((self._cc_centroids - rgb) ** 2, axis=1)
        return np.argmin(distance)

    def get_bc_RGB_from_tokens(self, tokens: np.ndarray) -> np.ndarray:
        try:
            rgb_colors = self._cc_centroids[tokens]
            if rgb_colors.ndim == 1:
                rgb_colors = rgb_colors.reshape(-1, 3)
            # noise = np.random.uniform(-10, 10, rgb_colors.shape)
            # rgb_colors_with_noise = rgb_colors + noise
            # rgb_colors_with_noise = np.clip(rgb_colors_with_noise, 0, 255)
            rgb_colors_rounded = np.rint(rgb_colors).astype(int)
            color_tuples = [tuple(color) for color in rgb_colors_rounded]
        except IndexError:
            logger.warning(f"IndexError: {tokens}")
            color_tuples = [(255, 255, 255)]
        return color_tuples

    def get_patch_from_token(
        self, token: Union[int, Tensor], cat: Union[int, Tensor], top_k: int = 20
    ) -> Optional[np.ndarray]:
        if isinstance(token, Tensor):
            token = token.item()
        if isinstance(cat, Tensor):
            cat = cat.item()
        image_prefix = os.path.join(self._config.data_root, "images/")
        cat_alias = "type" if "type" in self._dataset.features.keys() else "category"
        type_ = self._dataset.features[cat_alias].feature.int2str(cat)
        postfix_ = "png" if cat_alias == "type" else "jpg"
        if cat_alias == "type":
            group_id = type_
        else:
            group_id = (
                self._group_mapping["categories"][type_]
                if self._group_mapping["categories"].get(type_, None)
                else type_
            )
        try:
            cluster_fullimages = self._img_data[str(group_id)]["cluster2hash"][
                str(token)
            ]
        except KeyError:
            full_candidates = list(self._img_data[group_id]["hash2cluster"].keys())
            selected_image_ = random.choices(full_candidates)[0]
            selected_image = selected_image_.split(".")[0] + f".{postfix_}"
            return os.path.join(image_prefix, f"{group_id}/{selected_image}")
        if not cluster_fullimages:
            return None
        cluster_images = sorted(cluster_fullimages, key=lambda x: x["distance"])[:top_k]
        distances = np.array([img["distance"] for img in cluster_images])
        # probabilities = [1 / d for d in distances]
        # probabilities /= np.sum(probabilities)
        selected_image = random.choices(
            cluster_images, weights=self._softmax(distances), k=1
        )[0]
        return os.path.join(
            image_prefix, f"{group_id}/{selected_image['hash']}.{postfix_}"
        )

    def get_canvas_size_from_token(
        self, tokens: Union[np.ndarray, Tensor]
    ) -> Tuple[int, int]:
        if self._config.is_canvas_cluster:
            if isinstance(tokens, Tensor):
                tokens = tokens.cpu().numpy()
            cw_token, ch_token = tokens[1], tokens[2]
            cw_real, ch_real = int(self._cw_centroids[cw_token]), int(
                self._ch_centroids[ch_token]
            )
        else:
            cw_real = int(
                self._dataset.features["canvas_width"].int2str(tokens[1].item())
            )
            ch_real = int(
                self._dataset.features["canvas_height"].int2str(tokens[2].item())
            )
        return cw_real, ch_real

    def _softmax(self, distances, tau=0.1):
        adjusted_neg_exp_distances = np.exp(-distances / tau)
        return adjusted_neg_exp_distances / np.sum(adjusted_neg_exp_distances)

    @property
    def _bbox_cluster(self) -> list:
        return {key: getattr(self, f"_{key}_centroids") for key in ["x", "y", "w", "h"]}
