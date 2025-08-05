import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch import BoolTensor, Tensor
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import datasets
from src.preprocess import get_preprocess

# import src.model.clip_backbone as clip_backbone
from .helpers import util
from .helpers.svg_crello import SVGBuilder
from .helpers.cluster_store import get_cluster_store, ClusterStoreConfig

logger = logging.getLogger(__name__)
# depress the PIL logger
pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.WARNING)


class CLAYDataset(Dataset):
    """CLAY dataset with layouts as targets."""

    def __init__(
        self,
        dataset_cfg: DictConfig,
        datafile_raw_path: str,
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize the JSON dataset."""
        self._dataset_cfg = dataset_cfg
        self._elem_bounds = (dataset_cfg.lower_bound, dataset_cfg.upper_bound)
        self._max_samples = max_samples
        # init process
        self._init_device()
        # load json file
        self._dataset = self._load_arrow_dataset(datafile_raw_path)

        # # Randomly sample if max_samples is specified
        # if self._max_samples is not None and self._max_samples < len(self._dataset):
        #     indices = np.random.choice(
        #         len(self._dataset),
        #         size=self._max_samples,
        #         replace=False
        #     )
        #     self._dataset = self._dataset.select(indices)
        #     logger.info(f"Randomly sampled {self._max_samples} examples from dataset")

        if not dataset_cfg.with_patch:
            current_columns = list(self._dataset.features.keys())
            columns_to_remove_ = ["image", "preview", "retrieve"]
            columns_to_remove = list(
                set(columns_to_remove_).intersection(current_columns)
            )
            self._dataset = self._dataset.remove_columns(columns_to_remove)
            logger.info("Remove columns: {}".format(columns_to_remove))
        # init vocab
        if not self._dataset_cfg.is_canvas_cluster:
            self._dataset_cfg.canvas_width_cluster_N = len(
                self._dataset.features["canvas_width"].names
            )
            self._dataset_cfg.canvas_height_cluster_N = len(
                self._dataset.features["canvas_height"].names
            )
            logger.info(
                f"Don't cluster canvas, current N_cw/N_ch: [{self._dataset_cfg.canvas_width_cluster_N},{self._dataset_cfg.canvas_height_cluster_N}]"
            )
        self._categories = self._dataset.features["type"].feature.names
        self._int2cat = self._dataset.features["type"].feature.int2str
        self._max_samples = (
            len(self._dataset) if self._max_samples is None else self._max_samples
        )
        # build per patch cluster weights
        try:
            self._cluster_store = get_cluster_store(
                ClusterStoreConfig(
                    data_root=self._dataset_cfg.cache_root,
                    cfg_type="visual_only",
                    is_canvas_cluster=self._dataset_cfg.is_canvas_cluster,
                ),
                self._dataset,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ClusterStore. Reason: {e}")
            self._cluster_store = None
        self._init_vocab()
        self._preprocess = get_preprocess(dataset_cfg.patch_transform)
        self._with_patch = dataset_cfg.with_patch
        self._svg_render = SVGBuilder(
            self._cluster_store,
            self._special_token_name_to_id,
            self._dataset.features["type"].feature.int2str,
        )
        self.seq_data = []
        self.cavas_data: List[Dict[str, Any]] = []
        self.data_patches = []
        self.order_dics = {}
        logger.info("Begin Dataset Initialization...")
        if self._dataset_cfg.elem_order == "optimized":
            with open(self._dataset_cfg.extern_order_map, "rb") as f:
                self._extern_order_map = np.load(f, allow_pickle=True).item()
            logger.debug(
                f"Load external order map successfully at: {self._dataset_cfg.extern_order_map}"
            )
        else:
            self._extern_order_map = None
        if self._read_cache(self._dataset_cfg.use_cache):
            logger.info("Dataset Initialization Done(Cache).")
            return
        # bboxes = []
        for idx, example in tqdm(enumerate(self._dataset), total=len(self._dataset)):
            # if self._max_samples is not None and idx >= self._max_samples:
            #     break
            ann_box_ = np.stack(
                [example["left"], example["top"], example["width"], example["height"]],
                axis=1,
            ).astype(np.float32)
            # pos_valid = (0 <= ann_box_) & (ann_box_ < 1)
            # valid_mask = np.all(pos_valid, axis=1)
            # if not valid_mask.any():
            #     # filter out the image with no valid bbox
            #     continue
            if idx == 0 and "saliency_v3_sum" not in example.keys():
                logger.warning(
                    "No saliency order in dataset. Use default order to replace it."
                )
            ann_cat_ = np.array(
                [
                    self._json_category_id_to_contiguous_id[_id]
                    for _id in example["type"]
                ]
            )
            if self._check_elem_len(ann_box_) is False:
                # filter out the image with too many elements
                continue
            # continuous [0-1] ann_box to [0-self.N_bbox-1]
            # bboxes.append(np.vstack(ann_box_))
            ann_box = self._bbox_quantization(np.vstack(ann_box_))
            ann_cat = np.array(ann_cat_).reshape(-1, 1)
            per_patch_tensors = []
            per_patch_tokens = []
            try:
                for ele_id in range(len(ann_cat_)):
                    other_tok = self._cluster_store(
                        example, ele_id, example["node_id"][ele_id]
                    )
                    per_patch_tokens.append(
                        self._per_patch_tokenize(
                            ann_cat[ele_id], ann_box[ele_id], other_tok, ele_id
                        )
                    )
                    if self._with_patch:
                        per_patch_tensors.append(
                            self._preprocess(example["image"][ele_id])
                        )
            except Exception as e:
                continue
            tokens = np.array(per_patch_tokens)
            if self._with_patch:
                per_patch_tensors_ = torch.stack(per_patch_tensors)
            idx += 1
            # # DEPRECATED: Sort boxes (now sort in __getitem()__)
            # order_idx = self._get_order(example)
            # tokens = tokens[order_idx]
            if self._with_patch:
                # per_patch_tensors_ = torch.index_select(
                #     per_patch_tensors, 0, torch.tensor(order_idx.copy())
                # )
                full_image_tensor = self._preprocess(example["preview"]).unsqueeze(0)
                if self._dataset_cfg.mix_visual_feat:
                    per_patch_tensors = torch.concat(
                        [full_image_tensor, per_patch_tensors_], dim=0
                    )
                else:
                    per_patch_tensors = per_patch_tensors_
            canvas_attr = self._cluster_store._get_canvas(example)
            self.seq_data.append(tokens)
            self.cavas_data.append({"id": example["id"], "attr": canvas_attr})
            if self._with_patch:
                self.data_patches.append(per_patch_tensors)
            self.order_dics.update({example["id"]: self.__get_order(example)})
        if self._dataset_cfg.use_cache or self._dataset_cfg.save_cache:
            self._save_cache()
        logger.info("Dataset Initialization Done.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.seq_data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, torch.Tensor]]]:
        """Get a sample from the dataset.

        Args:
            idx (int): index of the sample provided by the iterator.

        Returns:
            tokens (torch.Tensor): tokens of the layout.
            patches (torch.Tensor): patches of the layout.
            mask (torch.Tensor): mask of the layout.
        """
        # Reorder the sample
        reordered_seq_data, reordered_data_patches = self._reorder_sample(idx)
        token_per_idx = torch.tensor(reordered_seq_data, dtype=torch.long)

        if (
            self._dataset_cfg.elem_order == "random"
            and self._dataset_cfg.shuffle_per_epoch
        ):
            token_per_idx = token_per_idx[torch.randperm(token_per_idx.size(0))]
        mask = token_per_idx != self.pad_token

        # Pad tokens
        padded_tokens = F.pad(
            token_per_idx,
            (0, 0, 0, self.max_ele_length - token_per_idx.size(0)),
            value=self.pad_token,
        )

        # Pad patches
        # We need to pad the second dimension, which represents the number of patches
        # The first dimension represents the global visual features.
        if self._with_patch:
            patch_per_idx = reordered_data_patches
            patches_max_elements = (
                self.max_ele_length + 1
                if self._dataset_cfg.mix_visual_feat
                else self.max_ele_length
            )
            padded_patches = torch.zeros((patches_max_elements, 3, 224, 224))
            padded_patches[: patch_per_idx.size(0)] = patch_per_idx
        else:
            padded_patches = torch.empty(0)
        # Update the mask for the padded tokens
        mask = F.pad(mask, (0, 0, 0, self.max_ele_length - mask.size(0)), value=False)
        # Canvas attributes
        canvas_attr_ = torch.tensor(self.cavas_data[idx]["attr"], dtype=torch.long)
        canvas_attr = self._add_offset_canvas(canvas_attr_)
        curr_id = self.cavas_data[idx]["id"]
        order_dic = self.order_dics[curr_id]
        if order_dic is not None:
            for key in order_dic:
                temp = torch.ones(self.max_ele_length, dtype=torch.long) * -1
                if isinstance(order_dic[key], np.ndarray):
                    order_values = torch.tensor(order_dic[key].copy(), dtype=torch.long)
                elif torch.is_tensor(order_dic[key]):
                    order_values = order_dic[key].clone().to(dtype=torch.long)
                else:
                    raise TypeError("Unsupported type for order_dic[key]")
                temp[: len(order_values)] = order_values
                order_dic[key] = temp
            order_dic["optimized"] = order_dic["neural"].clone()
        else:
            order_dic = {}
        return (
            padded_tokens,
            padded_patches,
            mask,
            {"id": self.cavas_data[idx]["id"], "attr": canvas_attr},
            order_dic,
        )

    def _bbox_quantization(
        self, bbox: np.array, image_W: float = 1440, image_H: float = 2560
    ) -> np.array:
        """Encode the bbox coordinates to x,y,w,h.

        Args:
            bbox (np.array): bbox coordinates.

        Returns:
            np.array: encoded bbox coordinates.
        """
        if self.bbox_quant_type == "uniform":
            return util.uniform_quantize_box(
                bbox, image_W, image_H, pow(2, self._dataset_cfg.precision)
            )
        elif self.bbox_quant_type == "cluster":
            return util.cluster_quantize_box(
                bbox,
                self._cluster_store._bbox_cluster,
                self._dataset_cfg.bbox_cluster_N,
                self._dataset_cfg.shared_bbox_vocab,
            )
        else:
            raise ValueError(
                "Bad shared_bbox_vocab: {}".format(self._dataset_cfg.shared_bbox_vocab)
            )

    def _init_vocab(self) -> None:
        # Now we have len(self.categories) colors for categories, and N_img_token colors for img token.
        self.colors = util.gen_colors(len(self._categories))
        self._special_token_name_to_id = {
            token: self.special_tokens.index(token)
            + self.N_category
            + self.N_bbox
            + self.N_zid
            + self.N_img_token
            + self.N_cavas
            for token in self.special_tokens
        }
        self._special_token_id_to_name = {
            v: k for (k, v) in self._special_token_name_to_id.items()
        }
        self._json_category_id_to_contiguous_id = {
            v: i + self.N_bbox
            for i, v in enumerate(
                [
                    self._dataset.features["type"].feature.str2int(c)
                    for c in self._categories
                ]
            )
        }

        self._contiguous_category_id_to_json_id = {
            v: k for k, v in self._json_category_id_to_contiguous_id.items()
        }
        logger.info("Vocab Size: <{}>".format(self.vocab_size))
        logger.info("Number of Special Tokens: <{}>".format(self.N_sp_token))
        logger.info("Number of Canvas Tokens: <{}>".format(self.N_cavas))
        logger.info("Number of Categories: <{}>".format(self.N_category))
        logger.info("Number of Bbox Tokens: <{}>".format(self.N_bbox))
        logger.info("Number of Z-index Tokens: <{}>".format(self.N_zid))
        logger.info("Number of Image Tokens: <{}>".format(self.N_img_token))

    def _init_device(self) -> None:
        if torch.cuda.is_available():
            self._device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self._device = torch.device("cpu")

    @property
    def tex_token(self) -> int:
        """Return the tex token indicating following text part."""
        return self._special_token_name_to_id["tex"]

    @property
    def ntx_token(self) -> int:
        """Return the ntx token indicating following non-text part."""
        return self._special_token_name_to_id["ntx"]

    @property
    def cav_token(self) -> int:
        """Return the cav token indicating following canvas attributes."""
        return self._special_token_name_to_id["cav"]

    @property
    def bos_token(self) -> int:
        """Return the bos token."""
        return self._special_token_name_to_id["bos"]

    @property
    def eos_token(self) -> int:
        """Return the eos token."""
        return self._special_token_name_to_id["eos"]

    @property
    def pad_token(self) -> int:
        """Return the pad token."""
        return self._special_token_name_to_id["pad"]

    @property
    def N_zid(self) -> int:
        if self._dataset_cfg.without_zid:
            return 0
        else:
            return self._dataset_cfg.upper_bound

    @property
    def N_layout(self) -> int:
        return self.N_bbox + self.N_category

    @property
    def N_layout_withz(self) -> int:
        return self.N_layout + self.N_zid

    @property
    def N_var_per_element(self) -> int:
        return self._dataset_cfg.N_var_per_element

    @property
    def N_sp_token(self) -> int:
        return len(self.special_tokens)

    @property
    def N_category(self) -> int:
        return len(self._categories)

    @property
    def N_img_token(self) -> int:
        return self._dataset_cfg.img_cluster_N

    @property
    def N_valid_token(self) -> int:
        return (
            self.N_bbox + self.N_zid + self.N_category + self.N_img_token + self.N_cavas
        )

    @property
    def N_cavas(self) -> int:
        return self.N_cavas_w + self.N_cavas_h + self.N_cavas_bc

    @property
    def N_cavas_w(self) -> int:
        return self._dataset_cfg.canvas_width_cluster_N

    @property
    def N_cavas_h(self) -> int:
        return self._dataset_cfg.canvas_height_cluster_N

    @property
    def N_cavas_bc(self) -> int:
        return self._dataset_cfg.canvas_color_cluster_N

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return (
            self.N_bbox
            + self.N_zid
            + self.N_category
            + self.N_img_token
            + self.N_sp_token
            + self.N_cavas
        )

    @property
    def special_tokens(self) -> List[str]:
        """Return the special tokens."""
        return self._dataset_cfg.special_tokens

    @property
    def bbox_quant_type(self) -> str:
        """Return the bbox quantization mode."""
        return self._dataset_cfg.bbox_quantization

    @property
    def N_bbox(self) -> int:
        """Return the size of the vocabulary."""
        if self.bbox_quant_type == "uniform":
            return pow(2, self._dataset_cfg.precision)
        elif self.bbox_quant_type == "cluster":
            if self._dataset_cfg.shared_bbox_vocab == "x-y-w-h":
                return self._dataset_cfg.bbox_cluster_N * 4
            elif self._dataset_cfg.shared_bbox_vocab == "xywh":
                return self._dataset_cfg.bbox_cluster_N
            else:
                raise ValueError(
                    "Bad shared_bbox_vocab: {}".format(
                        self._dataset_cfg.shared_bbox_vocab
                    )
                )
        else:
            raise ValueError("Bad quantization mode: {}".format(self.bbox_quant_type))

    def _cat_id2name(self, cat_id: int) -> str:
        json_id = self._contiguous_category_id_to_json_id[cat_id]
        return self._dataset.features["type"].feature.int2str(json_id)

    def _check_elem_len(self, element: np.ndarray) -> bool:
        is_valid = False
        if element.shape[0] == 0:
            # empty element
            return False
        if self._elem_bounds[0] <= len(element) <= self._elem_bounds[1]:
            is_valid = True
        else:
            is_valid = False
        return is_valid

    def _per_patch_tokenize(
        self,
        cat_tok: int,
        pos_tok: np.ndarray,
        other_tok: Dict[str, int],
        ele_id: int,
        is_pad: bool = True,
    ) -> np.ndarray:
        img_tok = other_tok["img"]
        per_patch_len = self.N_var_per_element
        is_text = self._cat_id2name(cat_tok.item()) in ["TEXT", "TEXT_INPUT", "LABEL"]
        if is_text:
            super_cat_arr = np.array(self.tex_token).reshape(-1)
        else:
            super_cat_arr = np.array(self.ntx_token).reshape(-1)
        if not self._dataset_cfg.without_zid:
            zid_token = np.array(ele_id + self._offset_tok["zid"]).reshape(-1)
        else:
            zid_token = np.array([], dtype=np.int32).reshape(-1)
        img_tok = np.array(img_tok + self._offset_tok["img"]).reshape(-1)
        patch_tok = np.concatenate(
            (super_cat_arr, cat_tok, pos_tok, zid_token, img_tok), axis=0
        )
        # If padding is required and the current length is less than N_var_per_element
        if is_pad and len(patch_tok) < per_patch_len:
            # Calculate how much padding is needed
            padding_length = per_patch_len - len(patch_tok)
            # Create a padding array with the appropriate length
            padding = np.zeros(padding_length, dtype=patch_tok.dtype) + self.pad_token
            # Concatenate the padding to the end of the patch_tok
            patch_tok = np.concatenate([patch_tok, padding])

        return patch_tok

    def _is_token_in_range(self, token: Tensor, key: str) -> BoolTensor:
        range_start = torch.tensor(self._range_tok_dic[key][0], device=token.device)
        range_end = torch.tensor(self._range_tok_dic[key][1], device=token.device)
        is_in_range = (range_start <= token) & (token < range_end)
        return is_in_range

    @property
    def _offset_tok(self) -> Dict:
        canvas_w_offset = self.N_layout_withz + self.N_img_token
        canvas_h_offset = canvas_w_offset + self.N_cavas_w
        offset = {
            "pos": 0,
            "cat": self.N_bbox,
            "zid": self.N_layout,
            "img": self.N_layout_withz,
            "canvas_w": canvas_w_offset,
            "canvas_h": canvas_h_offset,
            "canvas_bc": canvas_h_offset + self.N_cavas_h,
        }
        if self._dataset_cfg.without_zid:
            offset.pop("zid")
        return offset

    @property
    def _range_tok(self) -> List[Tuple[int, int]]:
        tok_split = (
            list(self._offset_pos.values())
            + list(self._offset_tok.values())[1:]
            + [self.N_valid_token]
            + [self.vocab_size]
        )
        return [(tok_split[i], tok_split[i + 1]) for i in range(len(tok_split) - 1)]

    @property
    def _range_tok_dic(self) -> Dict[str, Tuple[int, int]]:
        tok_dic = {
            "x": (
                self._offset_pos["x"],
                self._offset_pos["x"] + self._dataset_cfg.bbox_cluster_N,
            ),
            "y": (
                self._offset_pos["y"],
                self._offset_pos["y"] + self._dataset_cfg.bbox_cluster_N,
            ),
            "w": (
                self._offset_pos["w"],
                self._offset_pos["w"] + self._dataset_cfg.bbox_cluster_N,
            ),
            "h": (
                self._offset_pos["h"],
                self._offset_pos["h"] + self._dataset_cfg.bbox_cluster_N,
            ),
            "cat": (self._offset_tok["cat"], self._offset_tok["cat"] + self.N_category),
            "img": (
                self._offset_tok["img"],
                self._offset_tok["img"] + self.N_img_token,
            ),
            "canvas_w": (
                self._offset_tok["canvas_w"],
                self._offset_tok["canvas_w"] + self.N_cavas_w,
            ),
            "canvas_h": (
                self._offset_tok["canvas_h"],
                self._offset_tok["canvas_h"] + self.N_cavas_h,
            ),
            "canvas_bc": (
                self._offset_tok["canvas_bc"],
                self._offset_tok["canvas_bc"] + self.N_cavas_bc,
            ),
            "special": (
                self._offset_tok["canvas_bc"] + self.N_cavas_bc,
                self._offset_tok["canvas_bc"] + self.N_cavas_bc + self.N_sp_token,
            ),
        }
        if not self._dataset_cfg.without_zid:
            tok_dic["zid"] = (
                self._offset_tok["zid"],
                self._offset_tok["zid"] + self.N_zid,
            )

        return tok_dic

    @property
    def _offset_pos(self) -> Dict:
        return {
            key: value * self._dataset_cfg.bbox_cluster_N
            for key, value in util.KEY_MULT_DICT[
                self._dataset_cfg.shared_bbox_vocab
            ].items()
        }

    @property
    def _offset_image(self) -> Tuple[int, int]:
        return (self._offset_tok["img"], self._offset_tok["canvas_w"])

    @property
    def max_seq_length(self) -> int:
        # 3 for cavas-attr,2 for bos/eos.
        return self.max_ele_length * self.N_var_per_element + 3 + 2

    @property
    def max_ele_length(self) -> int:
        full_max_elem = max([len(tokens) for tokens in self.seq_data])
        if self._dataset_cfg.upper_bound is None:
            return full_max_elem
        else:
            return min(full_max_elem, self._dataset_cfg.upper_bound)

    def _get_layer_order(
        self, example: Any, is_ingroup_raster: bool = False
    ) -> np.ndarray:
        """Get layer-based order for CLAY dataset, with configurable intra-layer ordering.

        Rules:
        1. Text elements (TEXT, TEXT_INPUT, LABEL) appear above non-text elements
        2. Within each category group:
           - If is_ingroup_raster=False: Order by element area (smaller on top)
           - If is_ingroup_raster=True: Apply raster ordering (top to bottom, left to right)

        Args:
            example: Dataset example containing element information
            is_ingroup_raster: If False, elements within same group are ordered by area.
                              If True, applies raster ordering within each group.

        Returns:
            np.ndarray: Indices for layer ordering from bottom to top
        """
        N = example["length"]
        categories = np.array(example["type"])

        # Identify text and non-text elements
        text_categories = ["TEXT", "TEXT_INPUT", "LABEL"]

        # Create masks for text and non-text elements
        text_masks = []
        for i in range(N):
            cat_name = self._dataset.features["type"].feature.int2str(
                int(categories[i])
            )
            text_masks.append(cat_name in text_categories)
        text_mask = np.array(text_masks)

        # Get indices for text and non-text elements
        text_indices = np.where(text_mask)[0]
        nontext_indices = np.where(~text_mask)[0]

        # Process text elements
        if len(text_indices) > 0:
            if is_ingroup_raster:
                # Apply raster ordering within text elements
                text_lefts = np.array([example["left"][idx] for idx in text_indices])
                text_tops = np.array([example["top"][idx] for idx in text_indices])
                text_raster_indices = np.lexsort((text_lefts, text_tops))
                text_indices = text_indices[text_raster_indices]
            else:
                # Order text elements by area (small to large)
                text_areas = np.array(
                    [
                        example["width"][idx] * example["height"][idx]
                        for idx in text_indices
                    ]
                )
                text_area_indices = np.argsort(text_areas)
                text_indices = text_indices[text_area_indices]

        # Process non-text elements
        if len(nontext_indices) > 0:
            if is_ingroup_raster:
                # Apply raster ordering within non-text elements
                nontext_lefts = np.array(
                    [example["left"][idx] for idx in nontext_indices]
                )
                nontext_tops = np.array(
                    [example["top"][idx] for idx in nontext_indices]
                )
                nontext_raster_indices = np.lexsort((nontext_lefts, nontext_tops))
                nontext_indices = nontext_indices[nontext_raster_indices]
            else:
                # Order non-text elements by area (small to large)
                nontext_areas = np.array(
                    [
                        example["width"][idx] * example["height"][idx]
                        for idx in nontext_indices
                    ]
                )
                nontext_area_indices = np.argsort(nontext_areas)
                nontext_indices = nontext_indices[nontext_area_indices]

        # Combine the indices: non-text elements at bottom, text elements on top
        # Note: We're returning in bottom-to-top order
        layer_order = np.concatenate([nontext_indices, text_indices])

        return layer_order

    def __get_order(self, example: Any) -> Dict[str, np.ndarray]:
        N = example["length"]
        order_dic = {
            # order before neural sort is default order of dataset.
            "neural": np.array(range(N)),
            "random": np.random.permutation(N),
            "raster": np.lexsort((example["left"], example["top"])),
            "layer": self._get_layer_order(example, is_ingroup_raster=False),
            "layer_raster": self._get_layer_order(example, is_ingroup_raster=True),
        }
        if "saliency_v3_sum" in example.keys():
            scores = example["saliency_v3_sum"]
            # Descending order by score
            assert (
                len(scores) == N
            ), "Number of patches should be equal to number of tokens."
            scores_wo_nan = np.nan_to_num(scores, nan=-np.inf)
            order_dic["saliency"] = np.argsort(scores_wo_nan)[::-1]
        else:
            order_dic["saliency"] = order_dic["neural"]
        return order_dic

    def _save_cache(self):
        data = {
            "seq_data": self.seq_data,
            "cavas_data": self.cavas_data,
            "data_patches": self.data_patches,
            "order_dics": self.order_dics,
        }
        filtered_data = self._filter_by_N_samples(data, self._max_samples)
        if self._with_patch and self._dataset_cfg.element_order == "neural":
            feat_prefix = "_mix" if self._dataset_cfg.mix_visual_feat else "_local"
            trans_prefix = f"_{self._dataset_cfg.patch_transform}"
        else:
            feat_prefix = ""
            trans_prefix = ""
        cache_path = os.path.join(
            self._dataset_cfg.cache_root,
            f"cache/cache{feat_prefix}{trans_prefix}_data.pt",
        )
        torch.save(filtered_data, cache_path)
        logger.info(f"Save cache successfully at {cache_path}.")

    def _read_cache(self, use_cache: bool = True):
        if not use_cache:
            logger.info("Don't use cache.")
            return False
        else:
            if self._with_patch and self._dataset_cfg.element_order == "neural":
                feat_prefix = "_mix" if self._dataset_cfg.mix_visual_feat else "_local"
                trans_prefix = f"_{self._dataset_cfg.patch_transform}"
            else:
                feat_prefix = ""
                trans_prefix = ""
            cache_path = os.path.join(
                self._dataset_cfg.cache_root,
                f"cache/cache{feat_prefix}{trans_prefix}_data.pt",
            )
            if os.path.exists(cache_path):
                data = torch.load(cache_path)
                filtered_data = self._filter_by_N_elements(
                    data, self._dataset_cfg.upper_bound, self._dataset_cfg.lower_bound
                )
                filtered_data = self._filter_by_N_samples(
                    filtered_data, self._max_samples
                )
                self.seq_data = filtered_data["seq_data"]
                self.cavas_data = filtered_data["cavas_data"]
                self.data_patches = filtered_data["data_patches"]
                self.order_dics = filtered_data["order_dics"]
                logger.info(
                    f"Load cache and filter by bounds [{self._dataset_cfg.lower_bound},{self._dataset_cfg.upper_bound}] successfully."
                )
                logger.info(f"Cache path: {cache_path}")
                return True
            else:
                logger.info(f"Cache not found at {cache_path}.")
                return False

    def _reorder_sample(self, idx: int) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """Reorder a single sample based on the current configuration."""
        curr_id = self.cavas_data[idx]["id"]
        order_dic = (
            self._extern_order_map[curr_id]
            if self._dataset_cfg.elem_order == "optimized"
            else self.order_dics[curr_id]
        )
        if self._dataset_cfg.elem_order == "optimized":
            assert (
                self._extern_order_map is not None
            ), "_extern_order_map should be loaded."
            # compatible with old version order_dics.
            curr_order = "optimized" if "optimized" in order_dic.keys() else "neural"
        elif (
            self._dataset_cfg.elem_order == "neural"
            and self._dataset_cfg.shuffle_for_neural
        ):
            curr_order = "random"
        else:
            curr_order = self._dataset_cfg.elem_order

        curr_perm_idx = order_dic[curr_order]
        valid_perm_idx_ = curr_perm_idx[curr_perm_idx != -1]
        valid_perm_idx = (
            valid_perm_idx_.cpu().numpy()
            if isinstance(valid_perm_idx_, torch.Tensor)
            else valid_perm_idx_
        )
        reordered_seq_data = self.seq_data[idx][valid_perm_idx]
        reordered_data_patches = None
        if self._with_patch:
            reordered_data_patches = self.data_patches[idx][valid_perm_idx]
        return reordered_seq_data, reordered_data_patches

    def _detokenize(self, tokens: Tensor):
        assert isinstance(tokens, Tensor), "tokens should be Tensor"
        assert (
            tokens.ndim == 2
        ), "tokens should be a 2D Tensor with shape (B, max_seq_length)"

        B, seq_len = tokens.size()
        assert (
            seq_len == self.max_seq_length
        ), "token shape should be (B, max_seq_length)"

        # Remove [bos] tokens if needed
        if tokens[0, 0] == self.bos_token:
            tokens = tokens[:, 1:]

        # Filter out invalid tokens based on EOS
        invalids = self._filter_eos(tokens)
        mask_eos = ~invalids

        # Sequence mask
        mask_dic = {
            "seq": torch.cat(
                [torch.ones(B, 1, device=tokens.device, dtype=torch.bool), mask_eos],
                dim=1,
            )
        }
        # Valid tokens
        token_valid = [tokens[i][mask_eos[i]] for i in range(B)]

        # Canvas attributes
        canvas_attrs = (
            torch.zeros(B, 4, device=tokens.device, dtype=torch.long) + self.pad_token
        )
        for i in range(B):
            valid_idx = min(4, len(token_valid[i]))
            canvas_attrs[i][:valid_idx] = token_valid[i][:valid_idx]
        # canvas_attrs = torch.stack([t[:4] for t in token_valid])

        # Validity check for canvas attributes
        canvas_valid = (
            (canvas_attrs[:, 0] == self.cav_token)
            & (canvas_attrs[:, 1].cpu().unsqueeze(1) >= self._range_tok[-4][0])
            & (canvas_attrs[:, 1].cpu().unsqueeze(1) < self._range_tok[-4][1])
            & (canvas_attrs[:, 2].cpu().unsqueeze(1) >= self._range_tok[-3][0])
            & (canvas_attrs[:, 2].cpu().unsqueeze(1) < self._range_tok[-3][1])
            & (canvas_attrs[:, 3].cpu().unsqueeze(1) >= self._range_tok[-2][0])
            & (canvas_attrs[:, 3].cpu().unsqueeze(1) < self._range_tok[-2][1])
        ).all(dim=1)

        if not canvas_valid.all():
            invalid_canvas_indices = (~canvas_valid).nonzero(as_tuple=True)[0]
            for idx in invalid_canvas_indices:
                logger.debug(f"Wrong parsed canvas_attr: {canvas_attrs[idx]}")

        canvas_attrs = self._deoffset_canvas(canvas_attrs)

        # Re-padding tokens
        token_repad = []
        for i in range(B):
            valid_idx = min(4, len(token_valid[i]))
            token_repad.append(self._repad_token(token_valid[i][valid_idx:]))
        token_repad = torch.stack(token_repad)
        # token_repad = [self._repad_token(t[4:]) for t in token_valid]
        # token_repad = torch.stack(token_repad)

        # De-offset tokens
        token_deoffset = self._deoffset_token(token_repad)

        # Check validity
        token_valid_per_ele = self._checktoken(token_deoffset)

        # Masks
        ele_mask = token_valid_per_ele != self.pad_token
        # DEPRECATED: row_mask now is full of True.
        row_mask = torch.ones_like(token_valid_per_ele, dtype=torch.bool)

        unpad_tokens = F.pad(
            token_valid_per_ele,
            (0, 0, 0, self.max_ele_length - token_valid_per_ele.size(1)),
            value=self.pad_token,
        )

        mask_dic["elem"] = F.pad(
            ele_mask, (0, 0, 0, self.max_ele_length - ele_mask.size(1)), value=False
        )
        mask_dic["row"] = F.pad(
            row_mask, (0, 0, 0, self.max_ele_length - row_mask.size(1)), value=False
        )
        # except Exception as e:
        #     logger.error(f"Error in detokenize: {e}")
        #     unpad_tokens = torch.full((B, self.max_ele_length, self.N_var_per_element), self.pad_token)
        #     canvas_attrs = torch.full((B, 4), self.pad_token)
        #     mask_dic = {
        #         "seq": torch.full((B, self.max_seq_length), False, dtype=torch.bool),
        #         "elem": torch.full((B, self.max_ele_length), False, dtype=torch.bool),
        #         "row": torch.full((B, self.max_ele_length), False, dtype=torch.bool),
        #     }
        return unpad_tokens, canvas_attrs, mask_dic

    def _deoffset_token(self, tokens_: Tensor) -> Tensor:
        tokens = tokens_.clone()

        B = tokens.size(0)
        tex_mask = tokens[:, :, 0] == self.tex_token
        ntx_mask = tokens[:, :, 0] == self.ntx_token
        pad_mask = tokens[:, :, 0] == self.pad_token
        ntx_offset = torch.tensor(
            [0]
            + [self._offset_tok["cat"]]
            + list(self._offset_pos.values())
            + [self._offset_tok["img"]],
            device=tokens.device,
        )
        valid_mask = ~pad_mask
        # Subtract offset from valid tokens
        tokens[valid_mask] -= ntx_offset

        neither_mask = ~(tex_mask | ntx_mask | pad_mask)
        if neither_mask.any():
            logger.debug(f"Wrong parsed tokens: {tokens[neither_mask]}")

        return tokens

    def _deoffset_canvas(self, canvas: Tensor) -> Tensor:
        assert isinstance(canvas, Tensor), "canvas should be Tensor"
        assert canvas.size(1) == 4, "canvas shape should be (B, 4)"
        canvas[:, 1] -= self._offset_tok["canvas_w"]
        canvas[:, 2] -= self._offset_tok["canvas_h"]
        canvas[:, 3] -= self._offset_tok["canvas_bc"]
        return canvas

    def _checktoken(self, tokens: Tensor) -> Tensor:
        assert isinstance(tokens, Tensor), "token should be Tensor"
        assert (
            tokens.size(2) == self.N_var_per_element
        ), f"token shape should be (B, N, {self.N_var_per_element})"

        # Check category validity
        cat_valid = (0 <= tokens[:, :, 1]) & (tokens[:, :, 1] < self.N_category)

        # Check position validity
        if self._dataset_cfg.shared_bbox_vocab == "x-y-w-h":
            pos_valid = (0 <= tokens[:, :, 2:6]) & (
                tokens[:, :, 2:6] < self._dataset_cfg.bbox_cluster_N
            )
        elif self._dataset_cfg.shared_bbox_vocab == "xywh":
            pos_valid = (0 <= tokens[:, :, 2:6]) & (tokens[:, :, 2:6] < self.N_bbox)
        else:
            raise NotImplementedError
        pos_valid = pos_valid.all(dim=2)

        # Check image token validity
        img_valid = (0 <= tokens[:, :, 6]) & (tokens[:, :, 6] < self.N_img_token)

        # Combine all validity checks
        valid = cat_valid & pos_valid & img_valid

        # Create output tensor filled with padding tokens
        valid_tokens = torch.full_like(tokens, self.pad_token)

        # Copy valid tokens
        valid_tokens[valid] = tokens[valid]

        return valid_tokens

    def _filter_invalid_labels_and_bboxes(
        self, label: Tensor, bbox: Tensor
    ) -> BoolTensor:
        # If a set of tokens for an element is corrupted, discard the element
        label_valid = (0 <= label) & (label < self.N_category)
        bbox_valid = (0 <= bbox) & (bbox < self.N_bbox)
        bbox_valid = torch.all(bbox_valid, dim=-1)
        invalid = torch.logical_not(label_valid & bbox_valid)
        return invalid

    def _filter_eos(self, tokens: Tensor) -> BoolTensor:
        assert tokens.ndim == 2, "tokens should be a 2D Tensor with shape (B, seq_len)"

        B, seq_len = tokens.size()
        invalid = torch.zeros(B, seq_len, dtype=torch.bool, device=tokens.device)

        if "bos" in self.special_tokens and "eos" in self.special_tokens:
            for i in range(B):
                token = tokens[i]
                invalid[i] = torch.cumsum(token == self.eos_token, dim=0) >= 1
                first_eos_idx = (token == self.eos_token).nonzero(as_tuple=True)[0]
                if len(first_eos_idx) > 0:
                    invalid[i, first_eos_idx[0]] = False
        else:
            invalid.fill_(False)

        return invalid

    def _repad_token(self, token_valid: Tensor) -> Tensor:
        # Remove padding tokens if any
        valid_mask = token_valid != self.pad_token
        if not valid_mask.any():
            return torch.full(
                (self.max_ele_length, self.N_var_per_element), self.pad_token
            )

        # Get valid tokens only
        valid_tokens = token_valid[valid_mask]

        # Calculate number of complete elements
        n_elements = len(valid_tokens) // self.N_var_per_element

        # Reshape valid tokens to (N, N_var_per_element)
        valid_tokens = valid_tokens[: n_elements * self.N_var_per_element].reshape(
            -1, self.N_var_per_element
        )

        # Create output tensor filled with padding
        output = torch.full(
            (self.max_ele_length, self.N_var_per_element), self.pad_token
        )

        # Copy valid elements (handling case where we need to truncate)
        n_copy = min(len(valid_tokens), self.max_ele_length)
        output[:n_copy] = valid_tokens[:n_copy]

        return output

    def render(
        self, tokens: torch.Tensor, canvas_attr_: torch.LongTensor, is_gt: bool = False
    ) -> Tuple[Image.Image, Any, List[Image.Image]]:
        assert isinstance(canvas_attr_, torch.LongTensor), "tokens should be LongTensor"
        # canvas_attr = [int(i.item()) for i in canvas_attr_]
        canvas_attr = canvas_attr_
        try:
            canvas_attr[1], canvas_attr[2] = (
                self._cluster_store.get_canvas_size_from_token(canvas_attr_)
            )
        except Exception as e:
            logger.error(f"Error in converting canvas attributes: {e}")
            canvas_attr[1], canvas_attr[2] = self._dataset_cfg.render_ratio
        if is_gt:
            # skip patch process if is_gt
            return self._svg_render(tokens, canvas_attr)
        rendered_svg = self._svg_render(tokens, canvas_attr)
        return rendered_svg

    def _load_arrow_dataset(self, datafile_raw_path: str) -> datasets.Dataset:
        all_datasets = []
        # Assume each batch folder starts with 'batch_'
        batch_folders = [
            f for f in os.listdir(datafile_raw_path) if f.startswith("batch_")
        ]
        batch_folders.sort()  # Ensure correct order

        for batch_folder in batch_folders:
            dataset_path = os.path.join(datafile_raw_path, batch_folder)
            dataset_ = datasets.load_from_disk(dataset_path)
            all_datasets.append(dataset_)

        # Use concatenate_datasets to merge all datasets into one
        dataset = datasets.concatenate_datasets(all_datasets)
        return dataset

    def _filter_by_len(
        self, max_length: int, min_length: int = 0, data: Optional[Dict] = None
    ) -> Union[List[int], Dict]:
        """Filter dataset by sequence length constraints.

        Args:
            max_length: Maximum allowed sequence length
            min_length: Minimum allowed sequence length (default: 0)
            data: Optional dictionary containing dataset components. If None, filters self.seq_data

        Returns:
            If data is None: List of valid indices
            If data is provided: Filtered data dictionary
        """
        # Handle max_length bounds
        if max_length > self._elem_bounds[1]:
            logger.warning(
                f"max_length {max_length} is greater than the upper bound {self._elem_bounds[1]}"
            )
            max_length = self._elem_bounds[1]

        if data is None:
            # Original index-only filtering mode
            valid_indices = []
            for i, seq_tok in tqdm(
                enumerate(self.seq_data),
                total=len(self.seq_data),
                desc=f"Filtering dataset by length [{min_length}, {max_length}]",
            ):
                if min_length <= len(seq_tok) <= max_length:
                    valid_indices.append(i)
            return valid_indices

        else:
            # Full data filtering mode
            filtered_seq_data = []
            filtered_canvas_data = []
            filtered_data_patches = []
            filtered_data_order_dics = {}

            for idx, seq in tqdm(
                enumerate(data["seq_data"]),
                total=len(data["seq_data"]),
                desc=f"Filtering data by length [{min_length}, {max_length}]",
            ):
                if min_length <= len(seq) <= max_length:
                    filtered_seq_data.append(seq)
                    filtered_canvas_data.append(data["cavas_data"][idx])
                    if self._with_patch:
                        filtered_data_patches.append(data["data_patches"][idx])
                    cur_id = data["cavas_data"][idx]["id"]
                    filtered_data_order_dics.update(
                        {cur_id: data["order_dics"][cur_id]}
                    )

            return {
                "seq_data": filtered_seq_data,
                "cavas_data": filtered_canvas_data,
                "data_patches": filtered_data_patches,
                "order_dics": filtered_data_order_dics,
            }

    def _filter_by_N_elements(
        self, data: Dict, upper_bound: int, lower_bound: int
    ) -> Dict:
        """Filter dataset by number of elements. Wrapper around _filter_by_len."""
        return self._filter_by_len(
            max_length=upper_bound, min_length=lower_bound, data=data
        )

    def _filter_by_N_samples(self, data, max_samples):
        # If max_samples is None or larger than dataset size, return original data
        if max_samples is None or max_samples >= len(data["seq_data"]):
            return data

        # Randomly select indices
        indices = np.random.choice(
            len(data["seq_data"]), size=max_samples, replace=False
        )

        # Initialize filtered data
        filtered_seq_data = [data["seq_data"][i] for i in indices]
        filtered_canvas_data = [data["cavas_data"][i] for i in indices]
        filtered_data_patches = []
        filtered_data_order_dics = {}

        # Handle patches if needed
        if self._with_patch:
            filtered_data_patches = [data["data_patches"][i] for i in indices]

        # Update order dictionary
        for idx in indices:
            cur_id = data["cavas_data"][idx]["id"]
            filtered_data_order_dics[cur_id] = data["order_dics"][cur_id]

        return {
            "seq_data": filtered_seq_data,
            "cavas_data": filtered_canvas_data,
            "data_patches": filtered_data_patches,
            "order_dics": filtered_data_order_dics,
        }

    def _add_offset_canvas(self, canvas_attr: torch.Tensor) -> torch.Tensor:
        canvas_attr = (
            canvas_attr.clone()
        )  # Create a copy to avoid modifying the original tensor
        canvas_attr[0] = canvas_attr[0] + self._offset_tok["canvas_w"]
        canvas_attr[1] = canvas_attr[1] + self._offset_tok["canvas_h"]
        canvas_attr[2] = canvas_attr[2] + self._offset_tok["canvas_bc"]
        return torch.tensor([self.cav_token] + canvas_attr.tolist(), dtype=torch.long)

    def _filter_by_N_samples(self, data: Dict, max_samples: int) -> Dict:
        """Filter dataset by maximum number of samples."""
        total_samples = len(data["seq_data"])
        if max_samples >= total_samples:
            return data

        # Generate random indices
        indices = np.random.choice(total_samples, size=max_samples, replace=False)

        return {
            "seq_data": [data["seq_data"][i] for i in indices],
            "cavas_data": [data["cavas_data"][i] for i in indices],
            "data_patches": (
                [data["data_patches"][i] for i in indices] if self._with_patch else []
            ),
            "order_dics": {
                data["cavas_data"][i]["id"]: data["order_dics"][
                    data["cavas_data"][i]["id"]
                ]
                for i in indices
            },
        }
