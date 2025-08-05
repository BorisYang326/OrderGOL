from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


ROOT = f"{str(Path(__file__).parent.parent)}"
DATA_ROOT = f"{ROOT}/data"

SPECIAL_TOKEN_VOCABULARIES = ["pad", "bos", "eos", "msk", "tex", "ntx", "cav"]
CHOICES = {
    "shared_bbox_vocab": ["xywh", "x-y-w-h"],
    "elem_order": ["raster", "random", "saliency", "neural", "optimized", "lexical", "layer", "layer_raster"],
    "bbox_quantization": ["uniform", "kmeans"],
    # "bbox_quantization": ["linear", "kmeans", "percentile"],
}
FONT_IMG_BALANCED_FACTOR = 1 / 3
NON_VISUAL_NUM = 7


@dataclass
class DataConfig:
    """Configuration class for the Design Data, contain some hyper-parameters."""

    # for extern data path
    ext_path: str = f"{ROOT}/data"
    # for dataloader
    batch_size: int = 24
    num_workers: int = 4
    drop_last: bool = False
    # for smoother curve
    train_shuffle: bool = True
    # for dataset
    cache_root: str = "." #should be overridden by hydra
    # for sequence
    special_tokens: Tuple[str] = ("tex", "ntx", "cav", "bos", "eos", "pad")
    is_patch_batch: bool = False
    # max variable number per element, max{'[tex]-c-x-y-w-h-ff-fs-fc','[nxt]-c-x-y-w-h-u'}
    N_var_per_element: int = 10
    elem_order: str = "random"
    order_id: Optional[int] = None
    lower_bound: int = 1
    upper_bound: int = 20
    max_samples: Optional[int] = None
    ## for layout token
    bbox_quantization: str = "cluster"
    precision: int = 8
    shared_bbox_vocab: str = "x-y-w-h"
    bbox_cluster_N: int = 50  # number of clusters for position embedding
    ## for visual token
    patch_transform: str = "clip"  # preprocess for image patch
    img_cluster_N: int = 50  # number of clusters for image embedding
    font_size_cluster_N: int = 10  # number of clusters for font size embedding
    font_family_cluster_N: Optional[int] = None  # number of clusters for font family embedding
    font_color_cluster_N: int = 10  # number of clusters for font color embedding
    ## for canvas token
    canvas_width_cluster_N: int = 10  # number of clusters for canvas width category
    canvas_height_cluster_N: int = 10  # number of clusters for canvas height category
    canvas_color_cluster_N: int = 10  # number of clusters for canvas color category
    ## for rendering
    render_ratio: tuple[int, int] = (540, 960)
    svg_save_num: int = 50  # number of svg files to save
    ## for dataset cache
    use_cache: bool = True
    ## for curriculum learning
    warmup_upper_bound: Optional[int] = None
    curricum_step: int = 5
    ## for combined visual patches
    mix_visual_feat: bool = False
    extern_order_map: Optional[str] = None
    order_pretrain_cfgs: Optional[str] = None
    order_pretrain_flag: bool = False
    order_pretrain_path: Optional[str] = None
    order_sweep_dir: Optional[str] = None  # for eval mode
    order_selected: Optional[list] = None
    with_patch: bool = False
    shuffle_for_neural: bool = False
    shuffle_per_epoch: bool = True
    is_canvas_cluster: bool = True
    is_visual_only: bool = False
    without_zid: bool = False
    is_sublevel_lexical: bool = False
    render_partial: bool = False

    def __post_init__(self) -> None:
        """Post init configuration."""
        # advanced validation like choices in argparse
        for key in ["shared_bbox_vocab", "bbox_quantization", "elem_order"]:
            assert getattr(self, key) in CHOICES[key]


@dataclass
class TrainerConfig:
    """Configuration class for the Trainer, contain some hyper-parameters."""

    # dir setting
    log_dir: str
    ckpt_dir: str
    exp_name: str
    run_mode: str = "full"
    scorer_dir: Optional[str] = None
    model_dir: Optional[str] = None
    codebook_dir: Optional[str] = None
    samples_dir: Optional[str] = None
    # dataset parameters
    split_ratio: float = 0.8
    sal_overlap_flag: bool = True
    # training parameters
    gpu: List[int] = field(default_factory=lambda: [0])
    # optimization parameters
    alpha: float = 0.0  # weight for cross-class token loss
    beta: float = 0.01  # weight for score entropy loss
    ce_weight: float = 1.0  # weight for cross entropy loss
    visual_balance_factor: float = 1.0  # balancing weight for img token
    # training parameters
    max_epochs: int = 50
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    s_grad_norm_clip: Optional[float] = None
    m_grad_norm_clip: Optional[float] = None
    weight_decay: float = 1e-2  # only applied on matmul weights
    end2end: bool = True  # end-to-end training, If False, only train the sort module
    # scorer parameters
    finetune_ratio: float = 1.0
    scr_ema_decay: float = 0.0  # EMA decay for scorer, should between 0 to 1.
    model_ema_decay: float = 0.9999  # EMA decay for model, should between 0 to 1.
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay: bool = False
    warmup_iters: int = 0
    final_iters: int = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    save_model: bool = False
    # sort module settings
    permutation_loss_type = "ce"  # mse or cross_entropy or kl_divergence
    det_sort: bool = True
    tau: Optional[float] = None
    neural_sort_version: str = "neuralsort"
    noise_scale_factor: float = 1.0
    score_norm_scale: float = 1.0
    reg_margin: float = 1.0
    # for model load
    pretrained_mode: str = "none"
    # for sampling
    sample_num: Optional[int] = None
    sampling: str = "random"  # see ./src/sampling.py for options
    # below are additional parameters for sampling modes
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: float = 5
    # for evaluation
    eval_mode: str = "gen"  # see ./src/trainer.py for options
    order_vis: bool = False
    render_vis: bool = False
    save_svg: bool = True
    # for memory control
    pin_mem_flag: bool = True
    # for different updating step of model optimizer
    supd_mode: str = "fixed"
    supd_steps: int = 0
    supd_threshold: float = 0.0
    supd_patience: int = 0
    # for curriculum learning
    curricum_patience: int = 0
    loss_ema_decay: float = 0.0
    # for scores regulation
    s_margin: float = 5e2
    s_entropy: float = 0.0
    s_kl: float = 0.0
    s_l2: float = 0.0
    margin_ratio: float = 1e-2
    # debug
    skip_train_loop: bool = False
    sup_order_mode: str = "none"
    save_epoch_interval: Optional[int] = None
    seed: Optional[int] = None
    clip_embed_inject: bool = True
    inject_ratio: float = 1.0
    inject_mode: str = "normal"
    inject_without_font: bool = True
    seq_extract: str = "model"
    save_order_interval: int = 3
    gen_multi_ratio: float = 1.0
    manual_reduction: bool = False
    dvae_type: str = "nat"
    load_best: bool = False
    wandb_mode: str = "online"
    eval_fid_steplist: Optional[List[int]] = None
    eval_fid_count: Optional[int] = None
    burn_gt_cache: bool = True
    multi_gpu: bool = False
    visual_fid_extract: str = "mae"
    is_computed_gt_saliency_metrics: bool = False
    is_optimize_sequentially: bool = False
    opt_seq_windowsize: int = 10
    compute_fid: bool = True

@dataclass
class ScorerConfig:
    """base config for scorer.

    Args:
            vocab_size (int): vocabulary size for embedding.
            d_model (int): hidden dimension of embedding.
            number_layers (int): number of layers in transformer.
            heads (int): number of heads in transformer.
            attr_dim (int): attribute dimension for layout annotation.
            device (torch.device): current device.
            drop_out (float, optional): drop out rate. Defaults to 0.2.
            image_fuser_depth (int, optional): number of layers in image fuser. Defaults to 4.
            layout_fuser_depth (int, optional): number of layers in layout fuser. Defaults to 4.
            cls_head_depth (int, optional): number of layers in classification head. Defaults to 1.
            img_backbone_name (Optional[str], optional):
                model name of pretrained CLIP image encoder. Defaults to None.
            multi_level_embed (bool, optional):
                whether use local+global image embedding. Defaults to False.
    """

    vocab_size: int = 1000
    learning_rate: float = 2.5e-4
    ratio_img_backbone: float = 1e-2
    embed_size: int = 512
    trans_enc_layers: int = 4
    trans_enc_heads: int = 8
    device: str = "cuda"
    img_backbone_name: Optional[str] = None
    imgenc_weight_path: Optional[str] = None
    attr_dim: int = 5
    drop_out: float = 0.0
    norm_first: bool = True
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    fc_out: bool = False
    scale_factor: float = 1.0
    norm_out: bool = True
    channel_mode: str = "both"
    fuse_mode: str = "add"
    layout_shrink: int = 1
    shared_layout_codebook: bool = True
    visual_emb_mode: str = "both"
    mask_out_font: bool = False
    font_img_balance_factor: float = 1 / 3
    use_ema_extract: bool = False
    use_fuse_v2: bool = False
