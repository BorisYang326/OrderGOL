"""Trainer module for training the model.

This module contains the Trainer class
TrainerConfig configuration class for training a model.
"""

import logging
import math
import os
import io
import cairosvg
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import sample as randsample
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from ema_pytorch import EMA
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import BoolTensor, FloatTensor, Tensor, LongTensor
from torchvision import transforms
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from thop import profile, clever_format
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import src.sort as sort
import wandb
from src.configs import FONT_IMG_BALANCED_FACTOR, NON_VISUAL_NUM
from src.dataset.crello_dataset import CrelloDataset
from src.dataset.clay_dataset import CLAYDataset
from src.fid.visual.mae import MAE
from src.fid.seq.fidnet_design_ae import DesignAE
from src.metric import compute_generative_model_scores, compute_saliency_aware_metrics
from src.model.codebook import Codebook
from src.model.layout_transformer import LayoutTransformer
from src.preprocess import get_preprocess
from src.sampling import SAMPLING_CONFIG_DICT, sample
from src.scorer import Scorer
from src.saliency.basnet import saliency_detect, get_saliency_model

# import src.utils as utils
from src.utils import (
    CustomSchedulerWrapper,
    EMA_Smoothing_Monitor,
    SampleGenerator,
    calculate_total_grad_norm,
    generate_gen_samples,
    generate_gt_samples,
    gradient_check_histogram,
    load_pickle,
    logger_format_time,
    permutation_matrix_to_ind,
    save_and_convert_svg,
    save_pickle,
    process_single_image,
    gen_salmaps_to_samples,
)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
logger = logging.getLogger(__name__)


class CustomDataParallel(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Trainer:
    """Trainer class for training the model."""

    def __init__(
        self,
        model: LayoutTransformer,
        codebook: Codebook,
        scorer: Scorer,
        cfg: DictConfig,
        dataset: Union[CrelloDataset, CLAYDataset],
    ) -> None:
        """Initialize the Trainer class."""
        self.trainer_cfg = cfg.trainer
        self.scheduler_cfg = {
            "model": cfg.model_scheduler,
            "scorer": cfg.scorer_scheduler,
            "codebook": (
                cfg.codebook_scheduler
                if hasattr(cfg, "codebook_scheduler")
                else cfg.model_scheduler
            ),
        }
        self._cfg = cfg
        self.optmizer_cfg = cfg.optimizer
        self.dataset = dataset
        self.data_cfg = cfg.data
        self.log_dir = self.trainer_cfg.log_dir
        tf_log_dir = os.path.join(self.log_dir, "tfboard")
        self.tfwriter = SummaryWriter(tf_log_dir)
        # simply set the wandb exp name with the order.
        self.wandbwriter = self._set_wandb(cfg)
        # check curricum learning
        self.current_upper_bound = (
            self.data_cfg.warmup_upper_bound
            if self.data_cfg.warmup_upper_bound is not None
            else self.data_cfg.upper_bound
        )
        logger.info(f"Current dataset upper bound: {self.current_upper_bound}")
        self.train_loader, self.test_loader = self._set_split_datasets(
            self.current_upper_bound
        )
        self.model, self.codebook, self.scorer = model, codebook, scorer
        self._init_emas()
        self.model, self.codebook, self.scorer = self._load_checkpoint(
            model, codebook, scorer, self.trainer_cfg.pretrained_mode
        )
        self.smoother = EMA_Smoothing_Monitor(beta=self.trainer_cfg.loss_ema_decay)
        # wandb.init(mode="disabled")
        self.s_dis_per_epoch = {
            "train": [[] for _ in range(self.trainer_cfg.max_epochs + 1)],
            "test": [[] for _ in range(self.trainer_cfg.max_epochs + 1)],
        }
        self.scores_reg_dic = {
            # "alpha": self.trainer_cfg.s_margin,
            # "beta": self.trainer_cfg.s_entropy,
            # "gamma": self.trainer_cfg.s_kl,
            # "zeta": self.trainer_cfg.s_l2,
            "margin_ratio": self.trainer_cfg.margin_ratio,
        }
        self._init_device()
        self._is_train = self.model.training
        self._stage = "train" if self._is_train else "test"
        self._iters = {"train": 0, "test": 0}

        assert self.trainer_cfg.visual_fid_extract in [
            "mae",
            "dreamsim",
        ], "invalid visual fid extract method."
        if self.trainer_cfg.visual_fid_extract == "mae":
            self.visual_ae = MAE(
                pretrained_path=self.dataset._dataset_cfg.cache_root
            ).to(self.device)
        else:
            from dreamsim import dreamsim

            model, preprocess = dreamsim(pretrained=True)
            self.visual_ae = model.to(self.device)
            self.preprocess = preprocess
        self.design_ae = DesignAE(
            dataset.vocab_size, dataset.pad_token, dataset.max_seq_length
        ).to(self.device)
        self.design_ae.load_model(
            self.dataset._dataset_cfg.cache_root,
            self.trainer_cfg.visual_balance_factor,
            postfix=self.trainer_cfg.exp_name.split("_")[0],
        )
        self._gt_cache_dirs = None
        self._sample_num = (
            len(self.dataset)
            if self.trainer_cfg.sample_num is None
            else self.trainer_cfg.sample_num
        )
        self._init_eval_fid_steps()

    def train(self) -> float:
        """Train function containing the main training threads per epoch."""
        self.optimizers, self.schedulers = self._config_optim()
        logger.debug(
            "Using scheduler for scorer {}".format(
                self.scheduler_cfg["scorer"]["_target_"]
            )
        )
        logger.debug(
            "Using scheduler for model {}".format(
                self.scheduler_cfg["model"]["_target_"]
            )
        )
        best_loss = float("inf")
        # self.save_checkpoint()
        count_no_improve = 0
        ## debug
        # test_loss = self._run_epoch_test(0)
        if self.data_cfg.elem_order in ["neural"] and (not self._cfg.debug):
            self._extract_order(0)
        # logger.info("test_loss: {:.2f} at Epoch: {}".format(test_loss, 0))

        # Initialize metrics dictionary at the start of training
        self.metrics_log = {"model_stats": None, "fid_metrics": []}

        for epoch_ in range(self.trainer_cfg.max_epochs):
            epoch = epoch_ + 1
            epoch_start_time = time.time()
            # epoch from 1 to max_epochs
            if not self.trainer_cfg.skip_train_loop:
                self._run_epoch_train(epoch)
            # maybe test loss is meaningless for small dataset and generation tasks.
            if self.test_loader is not None:
                test_loss = self._run_epoch_test(epoch)
                logger.info("test_loss: {:.2f} at Epoch: {}".format(test_loss, epoch))
            else:
                test_loss = torch.inf

            if epoch in self.eval_fid_steps:
                # Get model stats on first FID evaluation
                if self.metrics_log["model_stats"] is None:
                    gflops, params = self._get_current_gflops()
                    self.metrics_log["model_stats"] = {
                        "gflops": gflops,
                        "params": params,
                    }

                score_seq, score_visual = self._eval_fid(
                    self._iters["train"], is_train_fid=True
                )

                # Record metrics
                metric_entry = {
                    "epoch": epoch_,
                    "steps": self._iters["train"],
                    "seq_fid": score_seq[0]["fid"],
                    "seq_coverage": score_seq[0]["coverage"],
                    "seq_density": score_seq[0]["density"],
                }

                if self.trainer_cfg.render_vis:
                    metric_entry.update(
                        {
                            "vis_render_fid": score_visual["render"][0]["fid"],
                            "vis_render_coverage": score_visual["render"][0][
                                "coverage"
                            ],
                            "vis_render_density": score_visual["render"][0]["density"],
                        }
                    )

                self.metrics_log["fid_metrics"].append(metric_entry)

                # Save metrics to JSON file
                metrics_path = os.path.join(self.log_dir, "scaling_metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(self.metrics_log, f, indent=2)

            # smoothing the test loss
            improve_flag = self.smoother.check_update(test_loss)
            best_loss = min(test_loss, best_loss) if improve_flag else best_loss
            if self.trainer_cfg.save_model:
                if self.trainer_cfg.ckpt_dir is not None and improve_flag:
                    logger.info(
                        "best_loss: {:.2f} at Epoch: {}".format(best_loss, epoch)
                    )
                    self._save_checkpoint(self.trainer_cfg.ckpt_dir, "best")
                if self.trainer_cfg.save_epoch_interval:
                    if epoch % self.trainer_cfg.save_epoch_interval == 0:
                        self._save_checkpoint(self.trainer_cfg.ckpt_dir, epoch, "epoch")
            epoch_duration = time.time() - epoch_start_time
            logger.debug(f"Epoch {epoch} took {logger_format_time(epoch_duration)}")
            # TODO: need new metric for curricum learning
            if self.data_cfg.warmup_upper_bound is not None:
                raise NotImplementedError("Curricum learning is not implemented.")
                if not improve_flag:
                    count_no_improve += 1
                else:
                    count_no_improve = 0
                if (
                    count_no_improve > self.trainer_cfg.curricum_patience
                    and self.current_upper_bound < self.data_cfg.upper_bound
                ):
                    _upper_bound = (
                        self.data_cfg.curricum_step + self.current_upper_bound
                    )
                    self.current_upper_bound = min(
                        _upper_bound, self.data_cfg.upper_bound
                    )
                    logger.info(
                        f"Increasing upper bound, adding more samples with upper bound <{self.current_upper_bound}>."
                    )
                    # reload the dataloader
                    self.train_loader, self.test_loader = self._set_split_datasets(
                        self.current_upper_bound
                    )
                    count_no_improve = 0
            if self.data_cfg.elem_order in ["neural"] and (
                epoch in [1] or epoch % self.trainer_cfg.save_order_interval == 0
            ):
                self._extract_order(epoch)
        # wandb.finish()
        self._save_checkpoint(self.trainer_cfg.ckpt_dir, "last")
        self._burn_gt_cache()
        return best_loss

    @torch.no_grad()
    def eval(self) -> None:
        """Test function containing the main testing threads."""
        if self.trainer_cfg.run_mode == "full":
            # load the best model
            self.model, self.codebook, self.scorer = self._load_checkpoint(
                self.model, self.codebook, self.scorer, "full-test"
            )

        sampling_cfg = OmegaConf.structured(
            SAMPLING_CONFIG_DICT[self.trainer_cfg.sampling]
        )
        OmegaConf.set_struct(sampling_cfg, False)
        if "temperature" in self.trainer_cfg:
            sampling_cfg.temperature = self.trainer_cfg.temperature
        if "top_p" in self.trainer_cfg and sampling_cfg.name == "top_p":
            sampling_cfg.top_p = self.trainer_cfg.top_p
        self.sampling_cfg = sampling_cfg

        # Compute saliency metrics
        if self._cfg.data.render_partial:
            self._compute_saliency_metrics(
                self._iters["train"], self.trainer_cfg.is_computed_gt_saliency_metrics
            )
            self.wandbwriter.finish()
            self.tfwriter.close()
            return

        # Continue regular evaluation
        assert (
            self._cfg.data.render_partial == False
        ), "Shouldn't render partial for regular evaluation."
        self._set_stage("eval")
        self._lazy_load_fidcfg(is_train_fid=False)

        # Compute FID scores
        score_seq, score_visual = self._eval_fid(self._iters["train"])

        self._burn_gt_cache()
        if self.trainer_cfg.compute_fid:
            gflops, params = self._get_current_gflops()
            logger.info(f"model GFLOPs: {gflops}")
            logger.info(f"model params: {params}")
            logger.info(
                f"seq-fid: {score_seq[0]['fid']:.2f}, seq-coverage: {score_seq[0]['coverage']:.2f}, seq-density: {score_seq[0]['density']:.2f}"
            )
            logger.info(
                f"vis-render-fid: {score_visual['render'][0]['fid']:.2f}, vis-render-coverage: {score_visual['render'][0]['coverage']:.2f}, vis-render-density: {score_visual['render'][0]['density']:.2f}"
            )
            self.wandbwriter.log(
                {
                    "model/GFLOPs": gflops,
                    "model/params": params,
                    "eval/seq-fid": score_seq[0]["fid"],
                    "eval/seq-coverage": score_seq[0]["coverage"],
                    "eval/seq-density": score_seq[0]["density"],
                    "eval/vis-render-fid": score_visual["render"][0]["fid"],
                    "eval/vis-render-coverage": score_visual["render"][0]["coverage"],
                    "eval/vis-render-density": score_visual["render"][0]["density"],
                }
            )
        self.wandbwriter.finish()
        self.tfwriter.close()

    def _run_epoch_train(self, epoch: int) -> None:
        self.model.train()
        self.codebook.train()
        self._set_stage("train")
        if self.data_cfg.elem_order == "neural":
            self.scorer.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        # Track which model to update in sequential optimization mode
        update_generator = True
        window_counter = 0

        with torch.autograd.set_detect_anomaly(True):
            for it, (seq, patches, mask, canvas_, order_dics) in pbar:
                self._iters[self._stage] += 1
                self._steps[self._stage] += 1
                seq, patches, mask, canvas, order_dics = self._set_input(
                    seq, patches, mask, canvas_, order_dics
                )
                loss, loss_ce = self._forward(
                    seq, patches, mask, canvas, epoch, order_dics
                )

                # Check if we're using sequential optimization
                sequential_mode = (
                    self.data_cfg.elem_order == "neural"
                    and self.trainer_cfg.is_optimize_sequentially
                )

                # Update window counter for sequential optimization
                if sequential_mode:
                    window_counter = (window_counter + 1) % (
                        2 * self.trainer_cfg.opt_seq_windowsize
                    )
                    update_generator = (
                        window_counter < self.trainer_cfg.opt_seq_windowsize
                    )

                # Zero gradients
                self.optimizers["model"].zero_grad()
                if self.data_cfg.elem_order == "neural":
                    self.optimizers["scorer"].zero_grad()

                # Backward pass
                loss.backward()

                # Calculate and log gradients
                total_norm_m = calculate_total_grad_norm(self.model.parameters())
                total_norm_c = calculate_total_grad_norm(self.codebook.parameters())
                self.wandbwriter.log(
                    {
                        "grads/model.total_norm": total_norm_m,
                        "grads/codebook.total_norm": total_norm_c,
                        f"steps/train": self._steps["train"],
                    }
                )

                # Update generator (model and codebook) if not in sequential mode or if it's generator's turn
                if not sequential_mode or update_generator:
                    if self.trainer_cfg.m_grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.trainer_cfg.m_grad_norm_clip
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self.codebook.parameters(),
                            self.trainer_cfg.m_grad_norm_clip,
                        )
                    self.optimizers["model"].step()

                # Update scorer if not in sequential mode or if it's scorer's turn
                if self.data_cfg.elem_order == "neural":
                    total_norm_s = calculate_total_grad_norm(self.scorer.parameters())
                    self.wandbwriter.log(
                        {
                            "grads/scorer.total_norm": total_norm_s,
                            f"steps/train": self._steps["train"],
                        }
                    )

                    if not sequential_mode or not update_generator:
                        if self.trainer_cfg.s_grad_norm_clip is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.scorer.parameters(),
                                self.trainer_cfg.s_grad_norm_clip,
                            )
                        self.optimizers["scorer"].step()

                # Update EMA models
                for key, ema_model in self.ema_models.items():
                    if ema_model is not None:
                        ema_model.update()

                # Extra scorer updates (only if not in simultaneous mode)
                upd_cnt = 0
                if self.data_cfg.elem_order == "neural" and not sequential_mode:
                    upd_cnt = self._extra_update_scorer(
                        seq, patches, mask, canvas, epoch, order_dics, loss.item()
                    )
                    self.tfwriter.add_scalar(
                        "scorer_update_count", upd_cnt, self._iters[self._stage]
                    )
                    self.wandbwriter.log(
                        {
                            "scores/update-count": upd_cnt,
                            f"iters/train": self._iters["train"],
                        },
                    )

                # Log which model is being updated in sequential mode
                if sequential_mode:
                    self.wandbwriter.log(
                        {
                            "sequential_opt/updating_generator": (
                                1 if update_generator else 0
                            ),
                            f"steps/train": self._steps["train"],
                        }
                    )

                lr_info = self._update_scheduler()
                lr_desc = ", ".join(
                    [f"{k.split('/')[-1]}::{v:.2e}" for k, v in lr_info.items()]
                )

                # Update description with sequential optimization info if needed
                seq_desc = ""
                if sequential_mode:
                    seq_desc = f" [{'GEN' if update_generator else 'ORD'}]"

                pbar.set_description(
                    f"epoch {epoch} iter {it}: loss {loss:.4f}.{seq_desc} lr <{lr_desc}>"
                )

        if self.data_cfg.elem_order == "neural":
            self.tfwriter.add_histogram(
                "train | scores distribution",
                np.array(self.s_dis_per_epoch["train"][epoch]),
                epoch,
            )
            self.wandbwriter.log(
                {
                    "distribution/train-scores": wandb.Histogram(
                        np.array(self.s_dis_per_epoch["train"][epoch])
                    ),
                    f"epochs/train": epoch,
                },
            )

    @torch.no_grad()
    def _run_epoch_test(self, epoch: int) -> float:
        self.model.eval()
        self.codebook.eval()
        self.model.to(self.device)
        self.codebook.to(self.device)
        if self.data_cfg.elem_order == "neural":
            self.scorer.eval()
            self.scorer.to(self.device)
        self._set_stage("test")
        losses = []
        losses_ce = []
        pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        for it, (seq, patches, mask, canvas_, order_dics) in pbar:
            # if it == 0 and epoch == 0:
            #     self._sample_data = (seq, patches, mask)
            seq, patches, mask, canvas, order_dics = self._set_input(
                seq, patches, mask, canvas_, order_dics
            )
            loss, loss_ce = self._forward(seq, patches, mask, canvas, epoch, order_dics)
            losses_ce.append(loss_ce)
            losses.append(loss.item())
            self._iters[self._stage] += 1
            pbar.set_description(f"epoch {epoch} iter {it}: loss {loss.item():.5f}")
        test_loss_ce = float(np.mean(losses_ce))
        test_loss = float(np.mean(losses))
        self.tfwriter.add_scalar(f"{self._stage} | loss-ce", test_loss_ce, epoch)
        self.tfwriter.add_scalar(f"{self._stage} | loss", test_loss, epoch)
        self.wandbwriter.log(
            {"test | loss": test_loss, "test | loss-ce": test_loss_ce}, step=epoch
        )
        if self.data_cfg.elem_order == "neural":
            self.tfwriter.add_histogram(
                "test | scores distribution",
                np.array(self.s_dis_per_epoch["test"][epoch]),
                epoch,
            )
            self.wandbwriter.log(
                {
                    "test | scores distribution": wandb.Histogram(
                        np.array(self.s_dis_per_epoch["test"][epoch])
                    )
                },
                step=epoch,
            )

        if (
            self.data_cfg.elem_order == "neural"
            and self.trainer_cfg.sup_order_mode != "none"
        ):
            return test_loss
        else:
            return test_loss_ce

    def _config_optim(
        self,
    ) -> Tuple[Dict[str, torch.optim.Optimizer], Dict[str, CustomSchedulerWrapper]]:
        optimizer = instantiate(self.optmizer_cfg)
        optimizer_dict = {
            "model": None,
            "scorer": None,
            "codebook": None,
        }
        scheduler_dict = {
            "model": None,
            "scorer": None,
            "codebook": None,
        }
        self.scorer.learning_rate = (
            self._cfg.scorer.learning_rate
            if self.data_cfg.elem_order == "neural"
            else 0
        )
        if self.trainer_cfg.end2end:
            # finetune for model.
            model_params = self.model.get_param_group(
                self.trainer_cfg,
            )
            codebook_params = self.codebook.get_param_group(self.trainer_cfg)
            optimizer_dict["model"] = optimizer(model_params)
            optimizer_dict["codebook"] = optimizer(codebook_params)
            scheduler_dict["model"] = CustomSchedulerWrapper(
                optimizer_dict["model"], self.scheduler_cfg["model"]
            )
            scheduler_dict["codebook"] = CustomSchedulerWrapper(
                optimizer_dict["codebook"], self.scheduler_cfg["codebook"]
            )
            if self.data_cfg.elem_order == "neural":
                scorer_params = self.scorer.get_param_group()
                optimizer_dict["scorer"] = optimizer(scorer_params)
                scheduler_dict["scorer"] = CustomSchedulerWrapper(
                    optimizer_dict["scorer"], self.scheduler_cfg["scorer"]
                )
            logger.debug("End-to-end training with scorer and original transformer.")
        else:
            scorer_params = self.scorer.get_param_group(self.trainer_cfg)
            optimizer_dict["scorer"] = optimizer(scorer_params)
            scheduler_dict["scorer"] = CustomSchedulerWrapper(
                optimizer_dict["scorer"], self.scheduler_cfg
            )
            logger.debug("Only train the scorer.")
        return optimizer_dict, scheduler_dict

    def _tokenize_tok(
        self,
        seq: Tensor,
        mask: BoolTensor,
        canvas: Tensor = None,
        is_autoreg: bool = True,
    ):
        B, S = seq.shape[:2]
        batch_seq = []
        batch_mask = []
        # +1 for bos token
        if is_autoreg:
            max_len = self.dataset.max_seq_length + 1
            bos = torch.tensor(self.dataset.bos_token, device=seq.device).unsqueeze(0)
            eos = torch.tensor(self.dataset.eos_token, device=seq.device).unsqueeze(0)
        else:
            max_len = self.dataset.max_seq_length
        for i in range(B):
            filter_seq = seq[i][mask[i]]
            canv = canvas[i]
            if is_autoreg:
                seq_with_specials = torch.cat([bos, canv, filter_seq, eos])
            else:
                seq_with_specials = torch.cat([canv, filter_seq])
            seq_packed = (
                torch.zeros(
                    max_len,
                    dtype=torch.long,
                    device=filter_seq.device,
                )
                + self.dataset.pad_token
            )
            mask_packed = torch.zeros(
                max_len,
                dtype=torch.bool,
                device=filter_seq.device,
            )
            seq_packed[: len(seq_with_specials)] = seq_with_specials
            mask_packed[: len(seq_with_specials)] = True
            batch_seq.append(seq_packed)
            batch_mask.append(mask_packed)
        return torch.stack(batch_seq), torch.stack(batch_mask)

    def _tokenize_emb(self, emb: FloatTensor, mask: BoolTensor, canvas: Tensor = None):
        assert (
            "bos" in self.dataset.special_tokens
            and "eos" in self.dataset.special_tokens
        ), "Need to add bos/eos token in the dataset."
        B, S = emb.shape[:2]
        batch_seq = []
        batch_mask = []
        pad_embed = self.codebook(
            torch.tensor(self.dataset.pad_token).to(emb.device)
        ).unsqueeze(0)
        eos_embed = self.codebook(
            torch.tensor(self.dataset.eos_token).to(emb.device)
        ).unsqueeze(0)
        bos_embed = self.codebook(
            torch.tensor(self.dataset.bos_token).to(emb.device)
        ).unsqueeze(0)
        emb = rearrange(emb, "B S (A H) -> B S A H", A=self.dataset.N_var_per_element)
        # +1 for auto-regressive x/y
        max_len = self.dataset.max_seq_length + 1
        for i in range(B):
            filter_emb = emb[i][mask[i]]
            # eos = repeat(eos_embed, "H -> 1 H")
            # bos = repeat(bos_embed, "H -> 1 H")
            cav_emb = self.codebook(canvas[i])
            emb_with_specials = torch.cat([bos_embed, cav_emb, filter_emb, eos_embed])
            emb_packed = (
                torch.full((max_len, 1), 0, device=filter_emb.device) + pad_embed
            )
            mask_packed = torch.zeros(
                max_len,
                dtype=torch.bool,
                device=filter_emb.device,
            )
            emb_packed[: len(emb_with_specials)] = emb_with_specials
            mask_packed[: len(emb_with_specials)] = True
            batch_seq.append(emb_packed)
            batch_mask.append(mask_packed)
        return torch.stack(batch_seq), torch.stack(batch_mask)

    def _get_classwise_mask(self, labels):
        mask = torch.ones(
            labels.size(0),
            labels.size(1),
            self.dataset.vocab_size,
            device=labels.device,
            dtype=torch.bool,
        )
        # Set the corresponding class range to False based on the value of each token
        for start, end in self.dataset._range_tok:
            # Use broadcasting to generate a mask for each class
            class_mask = (labels >= start) & (labels < end)
            # Expand on the third dimension to match vocab_size
            class_mask = class_mask.unsqueeze(-1).expand(
                -1, -1, self.dataset.vocab_size
            )
            # Use scatter_ to set the corresponding class part of the mask to False
            mask.scatter_(
                2,
                torch.arange(start, end, device=labels.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(labels.size(0), labels.size(1), -1),
                ~class_mask,
            )
        # Apply padding mask
        pad_mask = labels == self.dataset.pad_token
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, self.dataset.vocab_size)
        mask[pad_mask] = False
        return mask

    def _get_weight_mask(self):
        weight_mask = torch.ones(
            1,
            1,
            self.dataset.vocab_size,
            device=self.device,
        )
        # weight should not be less than 1
        if self._cfg.data.is_visual_only:
            img_weight = max(self.trainer_cfg.visual_balance_factor, 1.0)
            weight_mask[
                :, :, self.dataset._offset_image[0] : self.dataset._offset_image[1]
            ] = img_weight
        else:
            img_weight = max(self.trainer_cfg.visual_balance_factor, 1.0)
            font_weight = max(img_weight / 3, 1.0)
            weight_mask[
                :, :, self.dataset._offset_image[0] : self.dataset._offset_image[1]
            ] = img_weight
            weight_mask[
                :, :, self.dataset._offset_font[0] : self.dataset._offset_font[1]
            ] = font_weight

        return weight_mask

    def _calculate_cross_class_penalty(self, logits_log_prob, label_mask, pad_mask):
        # Apply the label mask to the log probabilities
        masked_log_prob = logits_log_prob * label_mask.float()

        # Calculate the valid token number (BS)
        valid_token_num = pad_mask.sum(dim=1, keepdim=True).float()
        valid_vocab_num_ = label_mask.sum(dim=-1).float()
        valid_vocab_num = valid_vocab_num_.masked_fill(valid_vocab_num_ == 0, 1.0)
        # Sum the masked log probabilities across the vocabulary axis
        cross_class_penalty_per_token = masked_log_prob.sum(dim=-1) / valid_vocab_num

        # Normalize by the valid count (effective class count) for each token
        cross_class_penalty_per_sample = (
            cross_class_penalty_per_token.sum(dim=-1, keepdim=True) / valid_token_num
        )

        # Average across the batch
        cross_class_penalty = cross_class_penalty_per_sample.mean()

        return cross_class_penalty

    @torch.no_grad()
    def _uncond_sample(self, batch_size: int = 1) -> torch.Tensor:
        init_seq = torch.full(
            (batch_size, self.dataset.max_seq_length),
            self.dataset.pad_token,
            dtype=torch.long,
            device=self.device,
        )
        init_seq[:, 0] = self.dataset.bos_token

        pad_mask = torch.full(
            (batch_size, self.dataset.max_seq_length),
            False,
            dtype=torch.bool,
            device=self.device,
        )
        pad_mask[:, 0] = True
        codebook_ = (
            self.ema_models["codebook"]
            if self.ema_models["codebook"] is not None
            else self.codebook
        )
        model_ = (
            self.ema_models["model"]
            if self.ema_models["model"] is not None
            else self.model
        )
        for step in range(1, self.dataset.max_seq_length):
            current_embeddings = codebook_(init_seq[:, :step])
            if self.trainer_cfg.clip_embed_inject:
                latest_tokens = init_seq[:, step - 1]
                for i in range(batch_size):
                    latest_token_ = latest_tokens[i]
                    if self.dataset._is_token_in_range(latest_token_, "img"):
                        last_cat_token = init_seq[i, step - 7]
                        try:
                            assert self.dataset._is_token_in_range(
                                last_cat_token, "cat"
                            ), f"last_cat_token:{last_cat_token}"
                            assert (
                                last_cat_token != self.dataset._offset_tok["cat"] + 1
                            ), f"last_cat_token:{last_cat_token}"
                        except AssertionError:
                            last_cat_token = torch.tensor(
                                self.dataset._offset_tok["cat"] + 2, device=self.device
                            )
                        deoff_last_cat_token = (
                            last_cat_token - self.dataset._offset_tok["cat"]
                        )
                        deoff_lastest_token = (
                            latest_token_ - self.dataset._offset_tok["img"]
                        )
                        retrieve_patch_path = (
                            self.dataset._cluster_store.get_patch_from_token(
                                deoff_lastest_token, deoff_last_cat_token
                            )
                        )
                        retrieve_patch = Image.open(retrieve_patch_path)
                        preprocess_patch = (
                            self.preprocess(retrieve_patch).unsqueeze(0).to(self.device)
                        )
                        clip_embedding_ = self.scorer.img_backbone(
                            preprocess_patch
                        ).unsqueeze(0)
                        clip_embedding = (
                            self.scorer.img_1x1conv(clip_embedding_.permute(0, 2, 1))
                            .permute(0, 2, 1)
                            .squeeze(0)
                        )
                        current_embeddings[i, -1] += clip_embedding

            # Generate logits for the next token
            logits_step = model_(current_embeddings, pad_mask[:, :step])

            # Sample the next token
            next_token = sample(logits_step[:, -1], self.sampling_cfg)
            init_seq[:, step] = next_token.squeeze(-1)

            # Update pad_mask to include the new token
            pad_mask[:, step] = True

        return init_seq

    def _init_device(self):
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU.")
            return

        # Get the environment variable for CUDA visible devices
        gpu_config_str = os.getenv("CUDA_VISIBLE_DEVICES")

        if gpu_config_str is None:
            logger.warning("CUDA_VISIBLE_DEVICES not set. Using all available GPUs.")
            gpu_config = list(range(torch.cuda.device_count()))
        else:
            gpu_config = list(map(int, gpu_config_str.split(",")))

        if len(gpu_config) == 0:
            raise ValueError("No GPUs available based on CUDA_VISIBLE_DEVICES setting.")

        # Always use the first GPU in the list as the primary device
        self.device = torch.device(f"cuda:{gpu_config[0]}")

        if len(gpu_config) > 1 and self.trainer_cfg.multi_gpu:
            # Use DataParallel if multiple GPUs are available and multi_gpu is enabled
            logger.info(f"Using DataParallel with GPUs: {gpu_config}")
            self._setup_data_parallel(gpu_config)
        else:
            # Use single GPU
            logger.info(f"Using single GPU: {gpu_config[0]}")
            # for single gpu(visible one),the gpu id is 0
            self.device = torch.device("cuda")
            self._to_device()

        logger.info(f"Using device: {self.device}")

    def _to_device(self):
        """Move models and components to the specified device"""
        self.model = self.model.to(self.device)
        self.scorer = self.scorer.to(self.device)
        self.scorer.device = self.device
        self.codebook = self.codebook.to(self.device)
        for key, ema_model in self.ema_models.items():
            if ema_model is not None:
                self.ema_models[key] = ema_model.to(self.device)

    def _setup_data_parallel(self, gpu_config):
        """Configure DataParallel for multiple GPUs"""
        self.model = CustomDataParallel(self.model, device_ids=gpu_config).to(
            self.device
        )
        self.scorer = CustomDataParallel(self.scorer, device_ids=gpu_config).to(
            self.device
        )
        self.codebook = CustomDataParallel(self.codebook, device_ids=gpu_config).to(
            self.device
        )
        self.scorer.device = self.device
        for key, ema_model in self.ema_models.items():
            if ema_model is not None:
                self.ema_models[key] = CustomDataParallel(
                    ema_model, device_ids=gpu_config
                ).to(self.device)

    def _load_checkpoint(
        self,
        model: LayoutTransformer,
        codebook: Codebook,
        scorer: Scorer,
        load_mode: str = "none",
    ):
        if load_mode == "none":
            logger.info("No pretrained model/codebook/scorer loaded.")
            return model, codebook, scorer
        model_path = self.trainer_cfg.model_dir
        codebook_path = self.trainer_cfg.codebook_dir
        scorer_path = self.trainer_cfg.scorer_dir
        load_prefix = "best" if self.trainer_cfg.load_best else "last"
        prefix_path = os.path.join(self.trainer_cfg.ckpt_dir, load_prefix)
        loaded_list = []
        if load_mode == "full-test":
            model_path = os.path.join(prefix_path, "model.pth")
            codebook_path = os.path.join(prefix_path, "codebook.pth")
            scorer_path = os.path.join(prefix_path, "scorer.pth")
        if load_mode in ["all", "full-test", "model_only", "model_codebook"]:
            try:
                model.load_state_dict(torch.load(model_path), strict=False)
                logger.info(f"Loading pretrained model from path: {model_path}")
                loaded_list.append("model")
            except FileNotFoundError:
                logger.warning(f"No pretrained model found at {model_path}")
        if load_mode in ["all", "full-test", "codebook_only", "model_codebook"]:
            try:
                codebook.load_state_dict(torch.load(codebook_path), strict=True)
                logger.info(f"Loading pretrained codebook from path: {codebook_path}")
                loaded_list.append("codebook")
            except FileNotFoundError:
                logger.warning(f"No pretrained codebook found at {codebook_path}")
        if load_mode in ["all", "full-test", "scorer_only"]:
            try:
                scorer.load_state_dict(torch.load(scorer_path), strict=False)
                logger.info(f"Loading pretrained scorer from path: {scorer_path}")
                loaded_list.append("scorer")
            except FileNotFoundError:
                logger.warning(f"No pretrained scorer found at {scorer_path}")

        for key, ema_model in self.ema_models.items():
            if ema_model is not None:
                ema_path = os.path.join(prefix_path, f"{key}_ema.pth")
                try:
                    if key in loaded_list:
                        self.ema_models[key] = torch.load(ema_path)
                        logger.info(
                            f"Loading pretrained {key}_ema from path: {ema_path}"
                        )
                except FileNotFoundError:
                    logger.warning(f"No pretrained {key}_ema found at {ema_path}")

        return model, codebook, scorer

    def _save_checkpoint(
        self,
        ckpt_dir_: str,
        pre_fix: str = "last",
        epoch: Optional[int] = None,
    ) -> None:
        """Save the model and scorer to the checkpoint directory."""
        logger.info("Saving model and scorer...")
        ckpt_dir = os.path.join(ckpt_dir_, pre_fix)
        os.makedirs(ckpt_dir, exist_ok=True)
        if epoch is not None:
            ckpt_dir = os.path.join(ckpt_dir, f"{epoch}")
        model_path = os.path.join(ckpt_dir, "model.pth")
        codebook_path = os.path.join(ckpt_dir, "codebook.pth")
        scorer_path = os.path.join(ckpt_dir, "scorer.pth")
        if self.data_cfg.elem_order == "neural" or self.trainer_cfg.clip_embed_inject:
            if self.trainer_cfg.end2end:
                torch.save(self.model.state_dict(), model_path)
                torch.save(self.codebook.state_dict(), codebook_path)
                torch.save(self.scorer.state_dict(), scorer_path)
                logger.info("model/codebook/scorer had been saved successfully.")
            else:
                torch.save(self.scorer.state_dict(), scorer_path)
                logger.info("scorer had been saved successfully.")
        else:
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.codebook.state_dict(), codebook_path)
            logger.info("model/codebook had been saved successfully.")
        for key, ema_model in self.ema_models.items():
            if ema_model is not None:
                ema_path = os.path.join(ckpt_dir, f"{key}_ema.pth")
                torch.save(ema_model, ema_path)
                logger.info(f"{key}_ema had been saved successfully.")

    @torch.no_grad()
    def _extract_features(self, samples, feature_type: str, seq_extract: str = "ae"):
        features = []
        if feature_type == "seq":
            # Extract seq-based features
            seq = samples[feature_type]["seq"]
            mask = samples[feature_type]["mask"]
            # Split the input into smaller batches
            seq_batches = torch.split(seq, self.data_cfg.batch_size)
            mask_batches = torch.split(mask, self.data_cfg.batch_size)
            for seq_batch, mask_batch in tqdm(
                zip(seq_batches, mask_batches),
                total=len(seq_batches),
                desc=f"Extracting {seq_extract}-seq-based features",
            ):
                seq_batch = seq_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                assert seq_extract == "ae", "seq encoder should be ae-based"
                feature = self.design_ae.extract_features(seq_batch, mask_batch)
                features.append(feature)
        elif feature_type == "visual":
            for img in tqdm(
                samples[feature_type]["img"],
                total=len(samples[feature_type]["img"]),
                desc="Extracting visual-based features",
            ):
                if img is None:
                    continue
                if self.trainer_cfg.visual_fid_extract == "mae":
                    feature = self.visual_ae.extract_embedding(
                        img.convert("RGB"), self.device
                    )
                else:
                    input_ = self.preprocess(img).to(self.device)
                    feature = self.visual_ae.embed(input_)
                features.append(feature)
        else:
            logger.warning("Unknown feature type: {}".format(feature_type))
            raise NotImplementedError
        # features should be a list,features[0] shape should be (B, D)
        if features[0].dim() == 2:
            features = torch.cat(features)
        else:
            features = torch.stack(features)
        return features

    def _render(
        self,
        samples: Dict[str, Dict[str, Tensor]],
        is_gt: bool,
        steps: Optional[int] = None,
    ):
        flag_ = "GT" if is_gt else "GEN"
        num_samples = len(samples["elem"]["seq"])
        pbar = tqdm(
            range(num_samples), desc=f"Rendering <{flag_}> SVG/IMGs", total=num_samples
        )

        rendered_svgs = []
        rendered_images = []
        rendered_layouts = []
        for i in pbar:
            svg, img, layout = process_single_image(
                i, samples, self.dataset, self._cfg.data.render_partial
            )
            rendered_svgs.append(svg)
            rendered_images.append(img)
            rendered_layouts.append(layout)

        # Rest of the method remains unchanged
        if self.trainer_cfg.save_svg:
            retain_indices = randsample(
                range(len(rendered_images)),
                min(self.data_cfg.svg_save_num, len(rendered_images)),
            )
            retain_svgs = [rendered_svgs[i] for i in retain_indices]
            retain_images = [
                Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode("utf-8"))))
                for svg in retain_svgs
            ]
            self.wandbwriter.log(
                {
                    f"render/{flag_}": [wandb.Image(img) for img in retain_images],
                    f"iters/{self._stage}": steps,
                }
            )
            os.makedirs(f"results/{flag_}/svg", exist_ok=True)
            os.makedirs(f"results/{flag_}/png", exist_ok=True)
            if is_gt:
                retain_ids = [samples["visual"]["id"][i] for i in retain_indices]
                img_prefix = "GT_"
            else:
                retain_ids = [i for i in range(len(retain_images))]
                img_prefix = f"GEN_step{steps}_"
            for i, (img, svg) in enumerate(zip(retain_images, retain_svgs)):
                img.save(f"results/{flag_}/png/{img_prefix}{retain_ids[i]}.png")
                with open(
                    f"results/{flag_}/svg/{img_prefix}{retain_ids[i]}.svg", "w"
                ) as f:
                    f.write(svg)
        return rendered_images, rendered_svgs, rendered_layouts

    def _eval_fid(self, steps: int, is_train_fid: bool = False):
        self._set_stage("eval")
        # Initialize sampling config and ground truth samples
        self._lazy_load_fidcfg(is_train_fid)
        steps_ = 1 if not is_train_fid else steps
        samples_gen = self._generate_samples(is_gt=False, steps=steps_)
        gt_size = len(self._gt_samples["seq"]["seq"])
        gen_size = len(samples_gen["seq"]["seq"])
        sample_iters = self._sample_num // self.data_cfg.batch_size + 1
        sample_cnt = min(sample_iters * self.data_cfg.batch_size, gt_size)
        if self.trainer_cfg.compute_fid:
            score_seq, score_visual = self._compute_fid(
                samples_gen, self._gt_samples, gen_size, sample_cnt, steps_
            )   
        else:
            score_seq, score_visual = None, None
        return score_seq, score_visual

    def _compute_features(self, samples: Dict[str, Any]):
        gt_seq_feats = self._extract_features(samples, "seq")
        gt_visual_feats = None
        if self.trainer_cfg.render_vis:
            gt_visual_feats = self._extract_features(samples, "visual")
        return {"seq": gt_seq_feats, "visual": gt_visual_feats}

    def _lazy_load_fidcfg(self, is_train_fid: bool = False):
        """Initialize FID configuration and ground truth samples."""
        stage = self._stage if not is_train_fid else "train"
        if not hasattr(self, "_fid_initialized") or not self._fid_initialized:
            # Set up sampling config
            sampling_cfg = OmegaConf.structured(
                SAMPLING_CONFIG_DICT[self.trainer_cfg.sampling]
            )
            OmegaConf.set_struct(sampling_cfg, False)
            if "temperature" in self.trainer_cfg:
                sampling_cfg.temperature = self.trainer_cfg.temperature
            if "top_p" in self.trainer_cfg and sampling_cfg.name == "top_p":
                sampling_cfg.top_p = self.trainer_cfg.top_p
            self.sampling_cfg = sampling_cfg

            # Load or generate ground truth samples and features
            self._gt_feats, self._gt_samples = self._load_gt_cache(self._sample_num)
            self._fid_initialized = True
            logger.info(f"Lazy load fid config and samples done in <{stage}> stage.")

    def _set_stage(self, stage: str):
        assert stage in ["train", "eval"], "Mode must be either 'train' or 'eval'"
        self._stage = stage
        models_to_set = [self.model, self.codebook, self.scorer]
        for model in models_to_set:
            if model is not None:
                if stage == "train":
                    model.train().to(self.device)
                else:
                    model.eval().to(self.device)
        if stage == "eval":
            # EMA models are always set to eval mode
            for key, ema_model in self.ema_models.items():
                if ema_model is not None:
                    ema_model.eval().to(self.device)

    def _permute(self, seq, patches, mask, canvas, epoch, order_dics, eval_mode=False):
        per_ele_mask = mask[..., 0]
        if eval_mode:
            if self._cfg.scorer.use_ema_extract:
                # use ema models for eval mode
                scorer_ = self.ema_models["scorer"]
                codebook_ = self.ema_models["codebook"]
                scorer_.eval()
                codebook_.eval()
                with torch.no_grad():
                    scores = scorer_(seq, mask, patches, self.codebook)
            else:
                with torch.no_grad():
                    scores = self.scorer(seq, mask, patches, self.codebook)
        else:
            scores = self.scorer(seq, mask, patches, self.codebook)
        perm, sort_error_rate, sup_order_loss, s_inputs = sort.return_permutation(
            scores,
            per_ele_mask,
            epoch,
            tau=self.trainer_cfg.tau,
            is_train=self._stage == "train",
            det_sort=self.trainer_cfg.det_sort,
            neural_sort_version=self.trainer_cfg.neural_sort_version,
            noise_scale_factor=self.trainer_cfg.noise_scale_factor,
            score_norm_scale=self.trainer_cfg.score_norm_scale,
        )
        if eval_mode:
            return (
                None,
                None,
                {
                    "perm": perm,
                    "s_inputs": s_inputs,
                    "s_raws": scores,
                    "ele_mask": per_ele_mask,
                },
                None,
            )
        self.tfwriter.add_scalar(
            f"{self._stage} | sort_error_rate",
            sort_error_rate,
            self._iters[self._stage],
        )
        self.wandbwriter.log(
            {
                "scores/sort_error_rate": sort_error_rate,
                f"steps/train": self._steps["train"],
            }
        )
        sparse_loss, _ = sort.scores_regulation(
            scores,
            per_ele_mask,
            scale=self.trainer_cfg.score_norm_scale,
            config=self.scores_reg_dic,
        )
        score_ind = permutation_matrix_to_ind(perm.clone().detach())
        new_seq_embed_ = self.codebook(seq)
        new_seq_embed = rearrange(new_seq_embed_, "B S A H -> B S (A H)")
        # assert torch.all(
        #     score_ind
        #     == torch.argsort(
        #         rearrange(scores, "B S 1 -> B (S 1)"), descending=True
        #     )
        # ), "score_ind should be the same as argsort(scores)."
        sorted_emb = torch.matmul(perm, new_seq_embed)
        # sorted_emb = torch.einsum('bij,bjk-> bik',perm,new_seq_embed)
        sorted_tok = torch.gather(
            seq, 1, repeat(score_ind, "B S -> B S A", A=seq.size(-1))
        )
        # sorted_tok_ = utils.emb2indices(rearrange(sorted_emb,"B S (A H) -> B (S A) H",A=9), self.codebook)
        # assert torch.allclose(
        #     sorted_emb,
        #     rearrange(self.codebook(sorted_tok), "B S A H -> B S (A H)"),
        #     atol=1e-6,
        # ), "sanity check fail, dismatch between emb and tok."
        sorted_mask = torch.gather(
            mask, 1, repeat(score_ind, "B S -> B S H", H=mask.size(-1))
        )
        pad_emb, _ = self._tokenize_emb(sorted_emb, sorted_mask, canvas)
        pad_tok, pad_mask = self._tokenize_tok(sorted_tok, sorted_mask, canvas)
        assert pad_emb.dim() == 3, "pad_emb should be B S H."
        assert pad_tok.dim() == 2, "pad_tok should be B S."
        x = pad_emb[:, :-1, :]
        y = pad_tok[:, 1:]
        # assert torch.allclose(
        #     pad_emb, self.codebook(pad_tok), atol=1e-6
        # ), "sanity check fail, dismatch between emb and tok."
        perm_dict = {
            "s_flatten": scores[per_ele_mask],
            "s_inputs": s_inputs,
            "s_raws": scores,
            "ele_mask": per_ele_mask,
            "perm": perm,
            "sparse_loss": sparse_loss,
            "sup_order_loss": sup_order_loss,
        }
        return x, y, perm_dict, pad_mask

    def _get_loss(self, x, y, other_loss: Dict[str, Tensor], pad_mask: BoolTensor):
        logits = self.model(x, pad_mask)
        logits_log_prob = F.log_softmax(logits, dim=-1)
        weight_mask = self._get_weight_mask()
        weighted_log_prob = logits_log_prob * weight_mask
        label_mask = self._get_classwise_mask(y)
        cross_class_penalty = self._calculate_cross_class_penalty(
            logits_log_prob, label_mask, pad_mask
        )
        loss_ce = F.nll_loss(
            weighted_log_prob.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=self.dataset.pad_token,
        )
        sparse_loss = other_loss["sparse_loss"]
        beta_scale = loss_ce.detach() / (1e-6 + torch.abs(sparse_loss.detach()))
        loss = (
            loss_ce * self.trainer_cfg.ce_weight
            + self.trainer_cfg.alpha * cross_class_penalty
            + self.trainer_cfg.beta * sparse_loss * beta_scale
        )
        self.tfwriter.add_scalar(
            f"{self._stage}/loss_cross_class",
            cross_class_penalty,
            self._iters[self._stage],
        )
        self.tfwriter.add_scalar(
            f"{self._stage}/loss_sparse",
            sparse_loss,
            self._iters[self._stage],
        )
        self.wandbwriter.log(
            {
                f"loss/loss_cross_penalty": cross_class_penalty,
                f"loss/loss_sparse": sparse_loss,
                f"steps/train": self._steps["train"],
            },
        )
        class_losses = {}

        for class_label, (start_idx, end_idx) in self.dataset._range_tok_dic.items():
            # dont calculate [special class loss] for pad.
            class_mask = (y >= start_idx) & (y <= end_idx) & pad_mask
            if class_mask.sum() == 0:  # Skip if this class does not exist in this batch
                continue
            class_logits = logits_log_prob[class_mask]
            class_labels = y[class_mask]

            class_loss = F.nll_loss(
                class_logits.reshape(-1, class_logits.size(-1)),
                class_labels.reshape(-1),
                ignore_index=self.dataset.pad_token,
                reduction="mean",  # Average the loss by class
            )

            class_losses[class_label] = class_loss.item()

            # Record to TensorBoard
            self.tfwriter.add_scalar(
                f"{self._stage} | classwise_loss_[{class_label}]",
                class_loss.item(),
                self._iters[self._stage],
            )
            self.wandbwriter.log(
                {
                    f"loss/loss-clswise-{class_label}": class_loss.item(),
                    f"steps/train": self._steps["train"],
                },
            )
        return loss, loss_ce.clone().detach().item()

    def _set_input(
        self,
        seq: Tensor,
        patches: Any,
        mask: Tensor,
        canvas_: Dict[str, Tensor],
        order_dics: Optional[Dict[str, Tensor]],
    ):
        B, S = seq.shape[:2]
        seq = seq.to(self.device)
        patches = patches.to(self.device)
        mask = mask.to(self.device)
        canvas = canvas_["attr"].to(self.device)
        if order_dics is not None:
            order_dics = {
                key: order_dics[key].to(self.device) for key in order_dics.keys()
            }
        return seq, patches, mask, canvas, order_dics

    def _forward(
        self,
        seq: Tensor,
        patches,
        mask: Tensor,
        canvas,
        epoch: int,
        order_dics: Optional[Dict[str, Tensor]],
    ):
        if self.data_cfg.elem_order == "neural":
            x, y, perm_dict, pad_mask = self._permute(
                seq, patches, mask, canvas, epoch, order_dics
            )
            other_loss = {
                "sparse_loss": perm_dict["sparse_loss"],
                "sup_order_loss": perm_dict["sup_order_loss"],
            }
            self.s_dis_per_epoch[self._stage][epoch].extend(
                perm_dict["s_flatten"].detach().cpu().flatten().tolist()
            )
        else:
            if self.trainer_cfg.clip_embed_inject:
                clip_embed = self.scorer._get_clip_embed(patches, mask)
                new_seq_embed_ = self._inject_clip_embed(seq, mask, clip_embed)
                new_seq_embed = rearrange(new_seq_embed_, "B S A H -> B S (A H)")
                pad_emb, _ = self._tokenize_emb(new_seq_embed, mask, canvas)
                pad_tok, pad_mask = self._tokenize_tok(seq, mask, canvas)
                x = pad_emb[:, :-1, :]
            else:
                if epoch == 0 and self._iters[self._stage] == 0:
                    logger.info("Do not inject visual embedding for non-neural case.")
                pad_tok, pad_mask = self._tokenize_tok(seq, mask, canvas)
                x = self.codebook(pad_tok[:, :-1])
            y = pad_tok[:, 1:]
            other_loss = {
                "sparse_loss": torch.tensor(0, device=self.device),
                "sup_order_loss": torch.tensor(0, device=self.device),
            }
        pad_mask_x = pad_mask[:, :-1]
        loss, loss_ce = self._get_loss(x, y, other_loss, pad_mask_x)
        self.wandbwriter.log(
            {
                f"loss/loss": loss,
                f"loss/loss_ce": loss_ce,
                f"steps/train": self._steps["train"],
            },
        )
        self.tfwriter.add_scalar(
            f"{self._stage}/loss_total",
            loss.item(),
            self._iters["train"],
        )
        self.tfwriter.add_scalar(
            f"{self._stage}/loss_ce",
            loss_ce,
            self._iters["train"],
        )
        return loss, loss_ce

    def _set_split_datasets(self, upper_bound: int) -> Tuple[DataLoader, DataLoader]:
        samples_indices = self.dataset._filter_by_len(upper_bound)
        len_samples = len(samples_indices)
        len_train = int(len_samples * self.trainer_cfg.split_ratio)
        len_test = len_samples - len_train
        self.iters_per_epoch = {
            "train": math.ceil(len_train / self.data_cfg.batch_size),
            "test": math.ceil(len_test / self.data_cfg.batch_size),
        }
        current_dataset = Subset(self.dataset, samples_indices)
        if self.trainer_cfg.split_ratio == 1.0:
            train_loader = DataLoader(
                current_dataset,
                shuffle=self.data_cfg.train_shuffle,
                pin_memory=self.trainer_cfg.pin_mem_flag,
                batch_size=self.data_cfg.batch_size,
                num_workers=self.data_cfg.num_workers,
                drop_last=self.data_cfg.drop_last,
            )
            logger.debug(
                f"Current train dataloader size: <{len(current_dataset)}>, test dataloader size: <N/A>."
            )
            return train_loader, None
        train_dataset, test_dataset = random_split(
            current_dataset,
            [len_train, len_test],
            torch.Generator().manual_seed(self.trainer_cfg.seed),
        )

        torch.save(
            {
                "train_indices": train_dataset.indices,
                "test_indices": test_dataset.indices,
            },
            "split_indices.pth",
        )

        train_loader = DataLoader(
            train_dataset,
            shuffle=self.data_cfg.train_shuffle,
            pin_memory=self.trainer_cfg.pin_mem_flag,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
            drop_last=self.data_cfg.drop_last,
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            pin_memory=self.trainer_cfg.pin_mem_flag,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
            drop_last=self.data_cfg.drop_last,
        )
        logger.debug(
            f"Current train dataloader size: <{len(train_dataset)}>, test dataloader size: <{len(test_dataset)}>."
        )
        return train_loader, test_loader

    def _inject_clip_embed(self, seq, mask, clip_embed):
        visual_mask = mask.clone()
        visual_mask[..., :NON_VISUAL_NUM] = False
        clip_embed_ = repeat(clip_embed, "B S H -> B S A H", A=seq.size(-1))
        is_image_mask_line = visual_mask[..., NON_VISUAL_NUM] & ~visual_mask[..., -1]
        is_image_mask = visual_mask & is_image_mask_line.unsqueeze(-1)
        is_text_mask_line = visual_mask[..., -1]
        if self.trainer_cfg.inject_without_font:
            is_image_mask_ = repeat(
                is_image_mask, "B S A -> B S A H", H=clip_embed.size(-1)
            )
            clip_embed_weighted = (
                clip_embed_ * is_image_mask_ * self.trainer_cfg.inject_ratio
            )
        else:
            banlance_weight = torch.ones_like(clip_embed_)
            banlance_weight = banlance_weight.masked_fill(
                repeat(
                    is_text_mask_line,
                    "B S -> B S A H",
                    A=seq.size(-1),
                    H=clip_embed.size(-1),
                ),
                FONT_IMG_BALANCED_FACTOR,
            )
            no_visual_mask = repeat(
                ~visual_mask.clone(), "B S A -> B S A H", H=clip_embed.size(-1)
            )
            # mask out non-visual embedding
            banlance_weight_novisual = banlance_weight.masked_fill_(no_visual_mask, 0)
            clip_embed_weighted = (
                clip_embed_ * banlance_weight_novisual * self.trainer_cfg.inject_ratio
            )
        if self.trainer_cfg.inject_mode == "stop_gradient":
            new_seq_embed_ = (
                self.codebook(seq) + clip_embed_weighted - clip_embed_weighted.detach()
            )
        elif self.trainer_cfg.inject_mode == "normal":
            new_seq_embed_ = self.codebook(seq) + clip_embed_weighted
        else:
            raise ValueError("Unknown inject_mode.")
        return new_seq_embed_

    def _extract_order(self, epoch_id: int):
        dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            pin_memory=self.trainer_cfg.pin_mem_flag,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
        )
        _pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Extracting order for epoch {epoch_id}",
        )
        order_pairs = {}
        for it, (seq, patches, mask, canvas_, order_dics) in _pbar:
            B = seq.size(0)
            seq, patches, mask, canvas, order_dics = self._set_input(
                seq, patches, mask, canvas_, order_dics
            )
            with torch.no_grad():
                _, _, perm_dict, _ = self._permute(
                    seq, patches, mask, canvas, epoch_id, order_dics, eval_mode=True
                )
                perm = perm_dict["perm"]
                s_inputs = perm_dict["s_inputs"]
                s_raws = perm_dict["s_raws"]
                score_ind = permutation_matrix_to_ind(perm.clone().detach())
                ele_mask = mask[..., 0]
                for b in range(B):
                    order_ = score_ind[b][ele_mask[b]]
                    valid_num = ele_mask[b].sum().item()
                    canvas_id = canvas_["id"][b]
                    random_ = order_dics["random"][b]
                    raster_ = order_dics["raster"][b]
                    saliency_ = order_dics["saliency"][b]
                    order_dics["optimized"][b][:valid_num] = order_
                    optimized_ = order_dics["optimized"][b]
                    s_inputs_b = s_inputs[b]
                    order_pairs[canvas_id] = {
                        "random": random_.cpu().numpy(),
                        "raster": raster_.cpu().numpy(),
                        "saliency": saliency_.cpu().numpy(),
                        "optimized": optimized_.cpu().numpy(),
                        "s_inputs": s_inputs_b,
                        "s_raws": s_raws[b][:valid_num],
                    }
        with open(
            os.path.join(self.log_dir, f"order_pairs_noinjected_{epoch_id}.npy"), "wb"
        ) as f:
            np.save(f, order_pairs)
        logger.info("order_pairs saved.")

    def _init_emas(self):
        self.ema_models = {"model": None, "codebook": None, "scorer": None}
        if self.trainer_cfg.scr_ema_decay > 0:
            self.ema_models["scorer"] = EMA(
                self.scorer,
                beta=self.trainer_cfg.scr_ema_decay,
                power=1,
                update_after_step=0,
                update_every=1,
            )
            logger.info(
                f"EMA for scorer with decay {self.trainer_cfg.scr_ema_decay} has been initialized."
            )
        if self.trainer_cfg.model_ema_decay > 0:
            self.ema_models["model"] = EMA(
                self.model,
                beta=self.trainer_cfg.model_ema_decay,
                power=1,
                update_after_step=0,
                update_every=1,
            )
            self.ema_models["codebook"] = EMA(
                self.codebook,
                beta=self.trainer_cfg.model_ema_decay,
                power=1,
                update_after_step=0,
                update_every=1,
            )
            logger.info(
                f"EMA for model/codebook with decay {self.trainer_cfg.model_ema_decay} has been initialized."
            )

    def _init_eval_fid_steps(self):
        # Set steps/epochs for fid evaluation in training loop.
        if self.trainer_cfg.eval_fid_steplist is not None:
            self.eval_fid_steps = self.trainer_cfg.eval_fid_steplist
        elif self.trainer_cfg.eval_fid_count is not None:
            step_size = math.ceil(
                self.trainer_cfg.max_epochs / self.trainer_cfg.eval_fid_count
            )
            self.eval_fid_steps = list(
                range(1, self.trainer_cfg.max_epochs + 1, step_size)
            )
        else:
            self.eval_fid_steps = []
        logger.info(
            f"Eval FID steps: {self.eval_fid_steps} of {self.trainer_cfg.max_epochs} epochs."
        )

    def _generate_samples(
        self, is_gt: bool, steps: Optional[int] = None
    ) -> Dict[str, Any]:
        if is_gt:
            generator = DataLoader(
                self.dataset,
                shuffle=True,
                pin_memory=self.trainer_cfg.pin_mem_flag,
                batch_size=self.data_cfg.batch_size,
                num_workers=self.data_cfg.num_workers,
            )
            samples = generate_gt_samples(
                generator,
                self._tokenize_tok,
                self.dataset._detokenize,
                self._sample_num,
            )
        else:
            generator = SampleGenerator(self._uncond_sample, self.data_cfg.batch_size)
            samples = generate_gen_samples(
                generator,
                self.dataset,
                int(self._sample_num * self.trainer_cfg.gen_multi_ratio),
            )
        if self.trainer_cfg.render_vis:
            (
                samples["visual"]["img"],
                samples["visual"]["svg"],
                samples["visual"]["layout"],
            ) = self._render(samples, is_gt, steps)
        return samples

    def _compute_fid(
        self,
        samples_gen: Dict[str, Dict[str, Tensor]],
        samples_gt: Dict[str, Dict[str, Tensor]],
        total_num: int,
        fid_num: int,
        steps: int,
    ):
        seq_score_seq = {"fid": -1.0, "coverage": -1.0, "density": -1.0}
        visual_render_score = {
            "render": {"fid": -1.0, "coverage": -1.0, "density": -1.0},
        }
        # Extract and calculate FID
        # Seq-based feature extraction and FID calculation logic...
        gt_seq_feats, gt_render_visual_feats = (
            self._gt_feats["seq"],
            self._gt_feats["visual"],
        )
        gen_seq_feats = self._extract_features(samples_gen, "seq")
        if self.trainer_cfg.render_vis:
            gen_visual_feats = self._extract_features(samples_gen, "visual")
        seq_list, visual_render_list = [], []
        multi_times = int(max(1, math.ceil(self.trainer_cfg.gen_multi_ratio)))
        logger.info(
            f"N_sample gen totally: {total_num}, N_sample for current fid computing: {fid_num}, {multi_times}x calculation for averaging."
        )
        for i in range(multi_times):
            indices = randsample(range(total_num), fid_num)
            seq_score_seq = compute_generative_model_scores(
                gt_seq_feats[:fid_num], gen_seq_feats[indices, :]
            )
            seq_list.append(seq_score_seq)

            if self.trainer_cfg.render_vis:
                fid_num_visual = min(fid_num, len(gt_render_visual_feats))
                if i == 0:
                    logger.info(f"Current fid_num for visual: {fid_num_visual}")
                indices_visual = randsample(
                    range(len(gen_visual_feats)), fid_num_visual
                )
                visual_render_score = compute_generative_model_scores(
                    gt_render_visual_feats[:fid_num_visual],
                    gen_visual_feats[indices_visual, :],
                )
                visual_render_list.append(visual_render_score)
        # Record the final FID scores
        metrics = {
            "seq-fid": seq_list,
            "seq-coverage": seq_list,
            "seq-density": seq_list,
        }

        for metric_name, metric_list in metrics.items():
            if "fid" in metric_name:
                values = [x["fid"] for x in metric_list]
            elif "coverage" in metric_name:
                values = [x["coverage"] for x in metric_list]
            elif "density" in metric_name:
                values = [x["density"] for x in metric_list]

            mean_value = np.mean(values)
            std_value = np.std(values)
            self.tfwriter.add_scalar(metric_name, mean_value, steps)
            self.wandbwriter.log(
                {
                    f"metric/{metric_name}": mean_value,
                    f"iters/{self._stage}": steps,
                }
            )
            if self.trainer_cfg.gen_multi_ratio > 1:
                std_value_str = f"±{std_value}"
            else:
                std_value_str = ""
            logger.info(
                f"{metric_name} score: {mean_value}{std_value_str} on step {steps}"
            )

        if self.trainer_cfg.render_vis:
            visual_metrics = {
                "visual_render-fid": [x["fid"] for x in visual_render_list],
                "visual_render-coverage": [x["coverage"] for x in visual_render_list],
                "visual_render-density": [x["density"] for x in visual_render_list],
            }

            for metric_name, values in visual_metrics.items():
                mean_value = np.mean(values)
                std_value = np.std(values)
                self.tfwriter.add_scalar(metric_name, mean_value, steps)
                self.wandbwriter.log(
                    {
                        f"metric/{metric_name}": mean_value,
                        f"iters/{self._stage}": steps,
                    }
                )
                if self.trainer_cfg.gen_multi_ratio > 1:
                    std_value_str = f"±{std_value}"
                else:
                    std_value_str = ""
                logger.info(
                    f"{metric_name} score: {mean_value}{std_value_str} on step {steps}"
                )
        else:
            # Add default values to lists when render_vis is False
            seq_list.append({"fid": -1.0, "coverage": -1.0, "density": -1.0})
            visual_render_list = [{"fid": -1.0, "coverage": -1.0, "density": -1.0}]

            # Set default values for metrics
            default_metrics = {
                "visual_render-fid": -1.0,
                "visual_render-coverage": -1.0,
                "visual_render-density": -1.0,
            }

            for metric_name, value in default_metrics.items():
                self.tfwriter.add_scalar(metric_name, value, steps)
                self.wandbwriter.log(
                    {
                        f"metric/{metric_name}": value,
                        f"iters/{self._stage}": steps,
                    }
                )
                logger.info(f"{metric_name} score: {value} on step {steps}")

        return seq_list, {"render": visual_render_list}

    def _load_gt_cache(self, sample_num: int, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(self.dataset._dataset_cfg.cache_root, "cache/")
        assert os.path.exists(cache_dir), f"No cache dir found at {cache_dir}"
        postfix = f"_{self.data_cfg.elem_order}"
        postfix += (
            f"_{self.data_cfg.order_id}" if self.data_cfg.order_id is not None else ""
        )
        feat_cache_path = os.path.join(
            cache_dir, f"gt_feats_N{sample_num}{postfix}.pkl"
        )
        samples_cache_path = os.path.join(
            cache_dir, f"gt_samples_N{sample_num}{postfix}.pkl"
        )
        self._gt_cache_dirs = [feat_cache_path, samples_cache_path]
        # Check feature cache
        gt_feat_cache = load_pickle(feat_cache_path)
        gt_samples_cache = load_pickle(samples_cache_path)
        if gt_feat_cache:
            logger.info(f"GT <feature> cache hit at {feat_cache_path}")
            assert (
                gt_samples_cache is not None
            ), "GT <samples> cache should hit if <feature> cache hits."
            return gt_feat_cache, gt_samples_cache

        # Check samples cache
        if gt_samples_cache:
            logger.info(f"GT <samples> cache hit at {samples_cache_path}")
        else:
            logger.info(
                f"GT <samples> cache miss, generating samples at {samples_cache_path}"
            )
            gt_samples_cache = self._generate_samples(is_gt=True)

        # Compute features
        gt_feat_cache = self._compute_features(gt_samples_cache)

        # Save caches
        save_pickle(feat_cache_path, gt_feat_cache)
        save_pickle(samples_cache_path, gt_samples_cache)
        logger.info(f"GT <feat+samples> cache generated and saved in {cache_dir}")
        if self.trainer_cfg.render_vis:
            assert (
                gt_feat_cache["visual"] is not None
            ), "gt visual features didn't load correctly for render_vis==True."
        return gt_feat_cache, gt_samples_cache

    def _burn_gt_cache(self):
        if self.trainer_cfg.burn_gt_cache or self.data_cfg.elem_order == "optimized":
            # for optimized order, we force to burn gt cache
            if self._gt_cache_dirs is not None:
                logger.info("Burning GT cache...")
                for cache_path in self._gt_cache_dirs:
                    if os.path.exists(cache_path):
                        try:
                            os.remove(cache_path)
                            logger.info(f"Removed cache file: {cache_path}")
                        except OSError as e:
                            logger.warning(
                                f"Error removing cache file {cache_path}: {e}"
                            )
                    else:
                        logger.warning(f"Cache file not found: {cache_path}")
            else:
                logger.warning("No GT cache dirs found.")

    def _set_wandb(self, cfg: DictConfig):
        # simply set the wandb exp name with the order.
        name_ = f"{self.trainer_cfg.exp_name}#{self.data_cfg.elem_order}"
        name_ += (
            f"_{self.data_cfg.order_id}" if self.data_cfg.order_id is not None else ""
        )
        writer = wandb.init(
            project="VisualOrder-Design-Generation",
            name=name_,
            # dir=self.log_dir,
            config=OmegaConf.to_container(cfg),
            save_code=True,
            settings=wandb.Settings(start_method="thread"),
            mode=self.trainer_cfg.wandb_mode,
        )
        # for key in ["train", "eval"]:
        # current only support train stage.
        for key in ["train"]:
            wandb.define_metric(f"iters/{key}")
            wandb.define_metric(f"steps/{key}")  # steps for schedulers
            wandb.define_metric(f"epochs/{key}")
            wandb.define_metric(f"distribution/*", step_metric=f"steps/{key}")
            wandb.define_metric(f"loss/*", step_metric=f"steps/{key}")
            wandb.define_metric(f"scores/*", step_metric=f"iters/{key}")
            wandb.define_metric(f"metric/*", step_metric=f"iters/{key}")
            wandb.define_metric(f"render/*", step_metric=f"iters/{key}")
        for key in ["model", "codebook", "scorer"]:
            wandb.define_metric(f"grads/{key}.*", step_metric=f"steps/train")
            wandb.define_metric(f"lr/{key}", step_metric=f"steps/train")
        self._iters = {"train": 0, "eval": 0}
        self._steps = {"train": 0, "eval": 0}
        return writer

    def _extra_update_scorer(
        self, seq, patches, mask, canvas, epoch, order_dics, initial_loss
    ):
        upd_cnt = 1
        patience_cnt = 0
        loss_prev = initial_loss

        while True:
            loss_curr, _ = self._forward(seq, patches, mask, canvas, epoch, order_dics)

            if self.trainer_cfg.supd_mode == "fixed":
                if upd_cnt > self.trainer_cfg.supd_steps:
                    break
            elif self.trainer_cfg.supd_mode == "adaptive":
                if loss_curr.item() - loss_prev > self.trainer_cfg.supd_threshold:
                    patience_cnt += 1
                    if patience_cnt > self.trainer_cfg.supd_patience:
                        break
                else:
                    patience_cnt = 0
            else:
                raise ValueError("Unknown scorer_update_mode.")

            self.optimizers["scorer"].zero_grad()
            self.optimizers["codebook"].zero_grad()
            loss_curr.backward()

            if self.trainer_cfg.s_grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.scorer.parameters(),
                    self.trainer_cfg.s_grad_norm_clip,
                )

            self.optimizers["scorer"].step()
            self.optimizers["codebook"].step()

            # update the steps for extra scorer update.
            self._steps[self._stage] += 1
            self._update_scheduler("scorer")
            self._update_scheduler("codebook")

            if self.ema_models["scorer"] is not None:
                self.ema_models["scorer"].update()
            self.ema_models["codebook"].update()

            loss_prev = loss_curr.item()
            upd_cnt += 1

        return upd_cnt

    def _update_scheduler(self, spec_key: Optional[str] = None):
        updated_keys = []

        if spec_key is not None:
            if spec_key in self.schedulers:
                self.schedulers[spec_key].step()
                updated_keys.append(spec_key)
        else:
            for key in self.schedulers:
                if self.schedulers[key] is not None:
                    if (
                        key == "scorer"
                        and self.schedulers[key].scheduler.get_last_lr()[0] <= 0
                    ):
                        continue
                    self.schedulers[key].step()
                    updated_keys.append(key)

        lr_info = {}
        for key in updated_keys:
            spec_model = getattr(self, key)
            lr = self.schedulers[key].scheduler.get_last_lr()[0]
            lr_info[f"lr/{key}_lr"] = lr

            self.wandbwriter.log(
                {
                    f"lr/{key}": lr,
                    f"steps/train": self._steps["train"],
                }
            )
            gradient_check_histogram(
                self.wandbwriter, spec_model, self._steps[self._stage], key
            )
        return lr_info

    def _get_current_gflops(self):
        seq, _, mask, canvas, _ = self.dataset[0]
        seq, mask = seq.to(self.device).unsqueeze(0), mask.to(self.device).unsqueeze(0)
        pad_tok, pad_mask = self._tokenize_tok(
            seq, mask, canvas["attr"].to(self.device).unsqueeze(0)
        )
        x = self.codebook(pad_tok[:, :-1])
        model_clone = deepcopy(self.model)
        macs_, params_ = profile(model_clone, inputs=(x, pad_mask[:, :-1]))
        gflops, params = clever_format([macs_ / 2, params_], "%.4f")
        return gflops, params

    def _compute_saliency_metrics(
        self, steps: int, is_computed_gt: bool = False
    ) -> None:
        """Compute saliency-aware metrics for generated samples and optionally ground truth samples.

        Args:
            steps: Current training iteration
            is_computed_gt: Whether to compute metrics for ground truth samples as well. Default is False.
        """
        # Generate samples with layout information (render_partial=True)
        samples_gen = self._generate_samples(is_gt=False, steps=steps)

        saliency_metrics = {}

        # Process generated samples first
        logger.info("Processing generated samples...")
        samples_gen = gen_salmaps_to_samples(samples_gen, self.device, is_gt=False)

        # Compute metrics for generated samples
        logger.info("Computing saliency-aware metrics for generated samples...")
        saliency_metrics["gen"] = compute_saliency_aware_metrics(
            samples_gen, self.dataset._dataset.features["type"].feature
        )

        # Log results for generated samples
        utilization = sum(saliency_metrics["gen"]["utilization"]) / len(
            saliency_metrics["gen"]["utilization"]
        )
        occlusion = sum(saliency_metrics["gen"]["occlusion"]) / len(
            saliency_metrics["gen"]["occlusion"]
        )
        readability = sum(saliency_metrics["gen"]["readability"]) / len(
            saliency_metrics["gen"]["readability"]
        )

        logger.info(
            f"Generated - utilization: {utilization:.4f}, occlusion: {occlusion:.4f}, readability: {readability:.4f}"
        )

        self.wandbwriter.log(
            {
                "eval/gen-utilization": utilization,
                "eval/gen-occlusion": occlusion,
                "eval/gen-readability": readability,
                "steps": steps,
            }
        )

        # Optionally compute metrics for ground truth samples
        if is_computed_gt:
            samples_gt = self._generate_samples(is_gt=True, steps=steps)

            logger.info("Processing ground truth samples...")
            samples_gt = gen_salmaps_to_samples(samples_gt, self.device, is_gt=True)

            logger.info("Computing saliency-aware metrics for ground truth samples...")
            saliency_metrics["gt"] = compute_saliency_aware_metrics(
                samples_gt, self.dataset._dataset.features["type"].feature
            )

            # Log results for ground truth samples
            utilization = sum(saliency_metrics["gt"]["utilization"]) / len(
                saliency_metrics["gt"]["utilization"]
            )
            occlusion = sum(saliency_metrics["gt"]["occlusion"]) / len(
                saliency_metrics["gt"]["occlusion"]
            )
            readability = sum(saliency_metrics["gt"]["readability"]) / len(
                saliency_metrics["gt"]["readability"]
            )

            logger.info(
                f"Ground Truth - utilization: {utilization:.4f}, occlusion: {occlusion:.4f}, readability: {readability:.4f}"
            )

            self.wandbwriter.log(
                {
                    "eval/gt-utilization": utilization,
                    "eval/gt-occlusion": occlusion,
                    "eval/gt-readability": readability,
                    "steps": steps,
                }
            )

        return saliency_metrics
