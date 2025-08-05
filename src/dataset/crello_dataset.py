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


class CrelloDataset(Dataset):
    """Crello dataset with layouts as targets."""

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
        ## for partial crello dataset
        # self._dataset = datasets.load_from_disk(datafile_raw_path)
        self._dataset = self._load_arrow_dataset(datafile_raw_path)
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
        # build per patch cluster weights
        try:
            self._cluster_store = get_cluster_store(
                ClusterStoreConfig(
                    data_root=self._dataset_cfg.cache_root,
                    cfg_type="full",
                    is_canvas_cluster=self._dataset_cfg.is_canvas_cluster,
                ),
                self._dataset,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ClusterStore. Reason: {e}")
            self._cluster_store = None
        # self._dataset_cfg.font_size_cluster_N = len(self._cluster_store._ff_data['ff2label'].keys())
        if self._dataset_cfg.font_family_cluster_N is None:
            self._dataset_cfg.font_family_cluster_N = len(
                self._cluster_store._ff_data["full_label2ff"].keys()
            )
        self._init_vocab()
        self._preprocess = get_preprocess(dataset_cfg.patch_transform)
        self._with_patch = dataset_cfg.with_patch
        self._svg_render = SVGBuilder(
            self._cluster_store,
            self._special_token_name_to_id,
            self._dataset.features["type"].feature.int2str,
            render_partial=dataset_cfg.render_partial,
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
        for idx, example in tqdm(enumerate(self._dataset), total=self._max_samples):
            if self._max_samples is not None and idx >= self._max_samples:
                break
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
            for ele_id in range(len(ann_cat_)):
                other_tok = self._cluster_store(example, ele_id)
                per_patch_tokens.append(
                    self._per_patch_tokenize(
                        ann_cat[ele_id], ann_box[ele_id], other_tok, ele_id
                    )
                )
                if self._with_patch:
                    per_patch_tensors.append(self._preprocess(example["image"][ele_id]))
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
            + self.N_txt_token
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
        logger.debug("Number of Special Tokens: <{}>".format(self.N_sp_token))
        logger.debug("Number of Canvas Tokens: <{}>".format(self.N_cavas))
        logger.debug("Number of Categories: <{}>".format(self.N_category))
        logger.debug("Number of Bbox Tokens: <{}>".format(self.N_bbox))
        logger.debug("Number of Z-index Tokens: <{}>".format(self.N_zid))
        logger.debug("Number of Image Tokens: <{}>".format(self.N_img_token))
        logger.debug("Number of Text Tokens: <{}>".format(self.N_txt_token))

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
    def N_txt_token(self) -> int:
        return (
            self._dataset_cfg.font_color_cluster_N
            + self._dataset_cfg.font_size_cluster_N
            + self._dataset_cfg.font_family_cluster_N
        )

    @property
    def N_valid_token(self) -> int:
        return (
            self.N_bbox
            + self.N_zid
            + self.N_category
            + self.N_img_token
            + self.N_txt_token
            + self.N_cavas
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
            + self.N_txt_token
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
        font_tok = [other_tok["ff"], other_tok["fs"], other_tok["fc"]]
        img_tok = other_tok["img"]
        per_patch_len = self.N_var_per_element
        zid_token = np.array(ele_id + self._offset_tok["zid"]).reshape(-1)
        # Determine if it's text based on the category token
        is_text = self._cat_id2name(cat_tok.item()) in ["textElement"]
        if is_text:
            assert (
                None not in font_tok
            ), "Font token should not contain None for text patches."
            font_tok = [
                other_tok["ff"] + self._offset_tok["ff"],
                other_tok["fs"] + self._offset_tok["fs"],
                other_tok["fc"] + self._offset_tok["fc"],
            ]
            tex_arr = np.array(self.tex_token).reshape(-1)
            font_tok = np.array(font_tok).reshape(-1)
            patch_tok = np.concatenate(
                (tex_arr, cat_tok, pos_tok, zid_token, font_tok), axis=0
            )
        else:
            assert (
                img_tok is not None
            ), "Image token should not be None for non-text patches."
            img_tok = np.array(img_tok + self._offset_tok["img"]).reshape(-1)
            nxt_arr = np.array(self.ntx_token).reshape(-1)
            patch_tok = np.concatenate(
                (nxt_arr, cat_tok, pos_tok, zid_token, img_tok), axis=0
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
        font_family_offset = self.N_layout_withz + self.N_img_token
        font_size_offset = font_family_offset + self._dataset_cfg.font_family_cluster_N
        font_color_offset = font_size_offset + self._dataset_cfg.font_size_cluster_N
        canvas_w_offset = font_color_offset + self._dataset_cfg.font_color_cluster_N
        canvas_h_offset = canvas_w_offset + self.N_cavas_w

        return {
            "pos": 0,
            "cat": self.N_bbox,
            "zid": self.N_layout,
            "img": self.N_layout_withz,
            "ff": font_family_offset,
            "fs": font_size_offset,
            "fc": font_color_offset,
            "canvas_w": canvas_w_offset,
            "canvas_h": canvas_h_offset,
            "canvas_bc": canvas_h_offset + self.N_cavas_h,
        }

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
        return {
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
            "zid": (self._offset_tok["zid"], self._offset_tok["zid"] + self.N_zid),
            "img": (
                self._offset_tok["img"],
                self._offset_tok["img"] + self.N_img_token,
            ),
            "ff": (
                self._offset_tok["ff"],
                self._offset_tok["ff"] + self._dataset_cfg.font_family_cluster_N,
            ),
            "fs": (
                self._offset_tok["fs"],
                self._offset_tok["fs"] + self._dataset_cfg.font_size_cluster_N,
            ),
            "fc": (
                self._offset_tok["fc"],
                self._offset_tok["fc"] + self._dataset_cfg.font_color_cluster_N,
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
        return (self._offset_tok["img"], self._offset_tok["ff"])

    @property
    def _offset_font(self) -> Tuple[int, int]:
        return (
            self._offset_tok["ff"],
            self._offset_tok["fc"] + self._dataset_cfg.font_color_cluster_N,
        )

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

    def __get_order(self, example: Any) -> torch.Tensor:
        N = example["length"]
        order_dic = {
            # order before neural sort is default order of dataset.
            "neural": np.array(range(N)),
            "random": np.random.permutation(N),
            "raster": np.lexsort((example["left"], example["top"])),
            "lexical": self._get_lexical_order(example),
            "layer": np.array(range(N)),
            "layer_raster": self._get_layer_raster_order(example),
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

    def _get_layer_order(
        self, example: Any, is_ingroup_raster: bool = False
    ) -> np.ndarray:
        """Get layer order with configurable intra-group ordering.

        Args:
            example: Dataset example containing element information
            is_ingroup_raster: If True, applies raster ordering within each group.
                             If False, uses the original layer order.

        Returns:
            np.ndarray: Indices for layer ordering
        """
        N = example["length"]
        
        # If not using raster ordering within groups, return the original order
        if not is_ingroup_raster:
            return np.array(range(N))
            
        # Otherwise, apply the group+raster ordering
        categories = np.array(example["type"])
        
        # Define category groups
        # 0: TXT, 1: SVG, 2: IMG, 3: BKG
        group_mapping = {
            0: 0,  # TXT -> group 0
            1: 1,  # SVG -> group 1
            2: 2,  # IMG -> group 2 (IMG+BKG group)
            3: 2,  # BKG -> group 2 (IMG+BKG group)
        }
        
        # Initialize with default layer order
        layer_order = np.array(range(N))
        
        # Group elements by category
        groups = {0: [], 1: [], 2: []}  # Three groups: TXT, SVG, IMG+BKG
        
        # Assign elements to groups
        for i in range(N):
            cat = categories[i]
            group_id = group_mapping[cat]
            groups[group_id].append(i)
        
        # Apply raster ordering within each group
        for group_id, group_indices in groups.items():
            if len(group_indices) > 1:
                # Extract positions for elements in this group
                group_left = np.array([example["left"][idx] for idx in group_indices])
                group_top = np.array([example["top"][idx] for idx in group_indices])
                
                # Apply raster ordering (top to bottom, left to right)
                raster_indices = np.lexsort((group_left, group_top))
                
                # Replace the original indices with raster-ordered indices
                reordered_group = [group_indices[idx] for idx in raster_indices]
                
                # Update the layer_order array
                for orig_idx, new_idx in zip(group_indices, reordered_group):
                    layer_order[orig_idx] = new_idx
        
        return layer_order

    def _get_layer_raster_order(self, example: Any) -> np.ndarray:
        """Get layer_raster order - groups elements by category (IMG+BKG, SVG, TXT),
        then applies raster ordering within each group.

        Args:
            example: Dataset example containing element information

        Returns:
            np.ndarray: Indices for layer_raster ordering
        """
        return self._get_layer_order(example, is_ingroup_raster=True)

    def _save_cache(self):
        data = {
            "seq_data": self.seq_data,
            "cavas_data": self.cavas_data,
            "data_patches": self.data_patches,
            "order_dics": self.order_dics,
        }
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
        torch.save(data, cache_path)
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
        # try:
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
        tex_offset_ = torch.tensor(
            [0]
            + [self._offset_tok["cat"]]
            + list(self._offset_pos.values())
            + [self._offset_tok["zid"]]
            + [self._offset_tok["ff"], self._offset_tok["fs"], self._offset_tok["fc"]],
            device=tokens.device,
        )

        ntx_offset_ = torch.tensor(
            [0]
            + [self._offset_tok["cat"]]
            + list(self._offset_pos.values())
            + [self._offset_tok["zid"]]
            + [self._offset_tok["img"]]
            + [0] * 2,
            device=tokens.device,
        )
        for i in range(B):
            tokens[i, tex_mask[i]] -= tex_offset_
            tokens[i, ntx_mask[i]] -= ntx_offset_
        # tokens[tex_mask] -= tex_offset_
        # tokens[ntx_mask] -= ntx_offset_

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
        ), "token shape should be (B, N_i, N_var_per_element)"
        B, N_i, N_var = tokens.size()
        tex_mask_ = tokens[:, :, 0] == self.tex_token
        ntx_mask_ = tokens[:, :, 0] == self.ntx_token

        cat_valid = (0 <= tokens[:, :, 1]) & (tokens[:, :, 1] < self.N_category)
        if self._dataset_cfg.shared_bbox_vocab == "x-y-w-h":
            pos_valid = (0 <= tokens[:, :, 2:6]) & (
                tokens[:, :, 2:6] < self._dataset_cfg.bbox_cluster_N
            )
        elif self._dataset_cfg.shared_bbox_vocab == "xywh":
            pos_valid = (0 <= tokens[:, :, 2:6]) & (tokens[:, :, 2:6] < self.N_bbox)
        else:
            raise NotImplementedError

        pos_valid = pos_valid.all(dim=2)
        zid_valid = (0 <= tokens[:, :, 6]) & (tokens[:, :, 6] < self.N_zid)
        img_valid = torch.full_like(cat_valid, False)
        ff_valid = torch.full_like(cat_valid, False)
        fs_valid = torch.full_like(cat_valid, False)
        fc_valid = torch.full_like(cat_valid, False)

        for i in range(B):
            token = tokens[i]
            ntx_mask = ntx_mask_[i]
            tex_mask = tex_mask_[i]
            img_valid[i][ntx_mask] = (0 <= token[ntx_mask, 7]) & (
                token[ntx_mask, 7] < self.N_img_token
            )
            ff_valid[i][tex_mask] = (0 <= token[tex_mask, 7]) & (
                token[tex_mask, 7] < self._dataset_cfg.font_family_cluster_N
            )
            fs_valid[i][tex_mask] = (0 <= token[tex_mask, 8]) & (
                token[tex_mask, 8] < self._dataset_cfg.font_size_cluster_N
            )
            fc_valid[i][tex_mask] = (0 <= token[tex_mask, 9]) & (
                token[tex_mask, 9] < self._dataset_cfg.font_color_cluster_N
            )

        valid = (
            cat_valid
            & pos_valid
            & zid_valid
            & (img_valid | (ff_valid & fs_valid & fc_valid))
        )
        valid_tokens = torch.full_like(tokens, self.pad_token)
        for i in range(B):
            valid_token = tokens[i][valid[i]]
            valid_tokens[i, : valid_token.size(0)] = valid_token

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

    def _repad_token(self, token_valid: Tensor):
        elements = []
        element = []
        for token in token_valid:
            if token.item() in [self.ntx_token, self.tex_token]:
                if element:
                    # valid element at least 8 tokens [ntx][cat][pos]*4[zid][img/ff/fs/fc]
                    if len(element) >= self.N_var_per_element - 2:
                        elements.append(element)
                element = [token.item()]
            elif token.item() == self.eos_token:
                break
            else:
                element.append(token.item())
        if element:
            # valid element at least 8 tokens [ntx][cat][pos]*4[zid][img/ff/fs/fc]
            if len(element) >= self.N_var_per_element - 2:
                elements.append(element)
        padded_elements = []
        for element in elements:
            padded_element = element[: self.N_var_per_element] + [self.pad_token] * (
                self.N_var_per_element - len(element)
            )
            padded_elements.append(padded_element)
        # max_elements = len(padded_elements)
        all_elements_tensor = torch.full(
            (self.max_ele_length, self.N_var_per_element), self.pad_token
        )
        if len(padded_elements) > self.max_ele_length:
            # logger.warning(f"Too many elements: {len(padded_elements)}")
            # just truncate the elements.
            padded_elements = padded_elements[: self.max_ele_length]
        for i, padded_element in enumerate(padded_elements):
            all_elements_tensor[i, : len(padded_element)] = torch.tensor(padded_element)
        return all_elements_tensor

    def render(
        self,
        tokens: torch.Tensor,
        canvas_attr_: torch.LongTensor,
        is_gt: bool = False,
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
        rendered_svg, layouts = self._svg_render(
            tokens, canvas_attr
        )
        return rendered_svg, layouts

    def _render(
        self, tokens: torch.Tensor, target_size: Tuple[int, int] = (540, 960)
    ) -> Image.Image:
        ##### DEPRECATED
        target_size = tuple(target_size)
        if tokens.nelement() == 0:
            logger.warning("Empty tokens provided, render nothing.")
            return Image.new("RGB", target_size, color=(169, 169, 169))
        # get mask
        tex_mask = tokens[:, 0] == self.tex_token
        ntx_mask = tokens[:, 0] == self.ntx_token
        if tex_mask.any():
            font_family_ttfs = self._cluster_store.get_ff_ttf_from_token(
                tokens[tex_mask, -3].detach().cpu().numpy()
            )
            font_size_pts = self._cluster_store.get_fs_pt_from_tokens(
                tokens[tex_mask, -2].detach().cpu().numpy()
            )
            font_colors = self._cluster_store.get_fc_RGB_from_tokens(
                tokens[tex_mask, -1].detach().cpu().numpy()
            )
        if ntx_mask.any():
            retr_imgfiles = self._cluster_store.get_patch_from_token(
                tokens[ntx_mask, -3].detach().cpu().numpy()
            )
            bkg_color = util.compute_avg_color(retr_imgfiles)
        else:
            bkg_color = (169, 169, 169)

        # get bbox, cat, font_family, font_size, font_color
        bbox_ = self._cluster_store.get_bbox_from_tokens(tokens[:, 2:6])
        bbox = util.denormalize_bbox(bbox_, target_size[0], target_size[1])
        cats = tokens[:, 1]
        # init canvas
        img = Image.new("RGB", target_size, color=bkg_color)
        draw = ImageDraw.Draw(img, "RGBA")
        ntx_idx = 0
        for i in range(len(tokens)):
            if tokens[i, 0] != self.tex_token:
                x1, y1, x2, y2 = util.convert_xywh_to_ltrb(bbox[i])
                cat = cats[i]
                col_cat = self.colors[cat]
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=tuple(col_cat) + (200,),
                    fill=tuple(col_cat) + (64,),
                    width=3,
                )
                img_to_paste = util.resize_and_crop_image(
                    retr_imgfiles[ntx_idx], x2 - x1, y2 - y1
                )
                alpha_mask = img_to_paste.split()[-1]
                img.paste(img_to_paste, (x1, y1), mask=alpha_mask)
                ntx_idx += 1
        tex_idx = 0
        for i in range(len(tokens)):
            if tokens[i, 0] == self.tex_token:
                x1, y1, x2, y2 = util.convert_xywh_to_ltrb(bbox[i])
                cat = cats[i]
                col_cat = self.colors[cat]
                font = ImageFont.truetype(
                    font_family_ttfs[tex_idx], size=font_size_pts[tex_idx]
                )
                fill_text = util.compute_fill_text(font, (x1, y1, x2, y2))
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=tuple(col_cat),
                    width=3,
                )
                draw.text(
                    (x1, y1), fill_text, fill=tuple(font_colors[tex_idx]), font=font
                )
                tex_idx += 1
        img = ImageOps.expand(img, border=2)
        return img

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

    def _add_offset_canvas(self, canvas_attr: torch.Tensor) -> torch.Tensor:
        canvas_attr = (
            canvas_attr.clone()
        )  # Create a copy to avoid modifying the original tensor
        canvas_attr[0] = canvas_attr[0] + self._offset_tok["canvas_w"]
        canvas_attr[1] = canvas_attr[1] + self._offset_tok["canvas_h"]
        canvas_attr[2] = canvas_attr[2] + self._offset_tok["canvas_bc"]
        return torch.tensor([self.cav_token] + canvas_attr.tolist(), dtype=torch.long)

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

    def _get_lexical_order(self, example: Any) -> np.ndarray:
        """Get lexical order that puts text elements first and non-text elements last.
        Within each group (text/non-text), elements are randomly ordered.

        Args:
            example: Dataset example containing type information

        Returns:
            np.ndarray: Indices for lexical ordering
        """
        N = example["length"]
        types = [
            self._dataset.features["type"].feature.int2str(t) for t in example["type"]
        ]
        if not self._dataset_cfg.is_sublevel_lexical:
            # Create masks for text and non-text elements
            text_mask = np.array([t == "textElement" for t in types])

            # Get indices for text and non-text elements
            text_indices = np.where(text_mask)[0]
            nontext_indices = np.where(~text_mask)[0]

            # Randomly shuffle within each group
            np.random.shuffle(text_indices)
            np.random.shuffle(nontext_indices)

            # Concatenate text indices first, then non-text indices
            order = np.concatenate([text_indices, nontext_indices])
        else:
            text_indices = np.where([t == "textElement" for t in types])[0]
            svg_indices = np.where([t == "svgElement" for t in types])[0]
            image_indices = np.where([t == "imageElement" for t in types])[0]
            bg_indices = np.where([t == "coloredBackground" for t in types])[0]
            np.random.shuffle(text_indices)
            np.random.shuffle(svg_indices)
            np.random.shuffle(image_indices)
            np.random.shuffle(bg_indices)
            nontext_groups = [svg_indices, image_indices, bg_indices]
            np.random.shuffle(nontext_groups)
            order = np.concatenate([text_indices] + nontext_groups)

        return order
