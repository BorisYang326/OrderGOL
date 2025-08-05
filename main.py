import logging
import os

import hydra
from hydra.core.config_store import ConfigStore
from hydra import compose
from hydra.utils import instantiate
from omegaconf import DictConfig
import gc
import subprocess
from src.configs import DataConfig, ScorerConfig, TrainerConfig
from src.scorer import Scorer
from src.model.codebook import Codebook
from src.trainer import Trainer
from src.utils import set_seed
from src.configs import ROOT
import torch
from torch.nn.parallel import DataParallel

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("src")
# logger.setLevel(logging.DEBUG)

cs = ConfigStore.instance()
cs.store(group="trainer", name="base_trainer_config", node=TrainerConfig)
cs.store(group="data", name="base_data_config", node=DataConfig)
cs.store(group="scorer", name="base_scorer_config", node=ScorerConfig)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig) -> None:
    if os.getenv("PUBLIC_PATH"):
        cfg.data.ext_path = os.getenv("PUBLIC_PATH")
        logger.info(f"current PUBLIC_PATH: {cfg.data.ext_path}")
    else:
        # ext_path for cross-platform/server running consistently
        logger.warning("Please set PUBLIC_PATH in config.py")
    if os.getenv("IS_WANDB_CONNECTED"):
        assert cfg.trainer.wandb_mode in [
            "online",
            "offline",
            "disabled",
        ], "Wrong wandb_mode."
        if cfg.trainer.wandb_mode != "disabled":
            if os.getenv("IS_WANDB_CONNECTED") == "1":
                cfg.trainer.wandb_mode = "online"
            else:
                cfg.trainer.wandb_mode = "offline"
    else:
        logger.warning("Please set IS_WANDB_CONNECTED flag manually.")
    if cfg.seed is not None:
        # set seed for reproducibility
        set_seed(cfg.seed)
        cfg.trainer.seed = cfg.seed

    # CHECK if use order pretrain
    if (
        cfg.data.elem_order == "optimized"
        and cfg.trainer.run_mode in ["train", "full"]
        and cfg.data.order_pretrain_flag
    ):
        # initialize pre trainer
        cfg_ = compose(
            config_name="main.yaml",
            overrides=cfg.data.order_pretrain_cfgs.split(" "),
        )
        cfg_.trainer.seed = cfg.seed
        cfg_.trainer.log_dir = "./order-pretrain"
        cfg_.trainer.ckpt_dir = "./order-pretrain/checkpoints"
        cfg_.data.ext_path = cfg.data.ext_path
        cfg_.trainer.wandb_mode = cfg.trainer.wandb_mode
        cfg_.trainer.order_saving_list = cfg.trainer.order_saving_list
        dataset = instantiate(cfg_.dataset)(
            dataset_cfg=cfg_.data,
        )

        scorer = Scorer(cfg_.scorer, dataset.vocab_size, cfg_.data.mix_visual_feat)
        codebook = Codebook(dataset.vocab_size, cfg_.model.n_embd, dataset.pad_token)
        model = instantiate(cfg_.model)(
            vocab_size=dataset.vocab_size, block_size=dataset.max_seq_length
        )
        scheduler_dict = {
            "model": cfg_.model_scheduler,
            "scorer": cfg_.scorer_scheduler,
        }
        pre_trainer = Trainer(
            model,
            codebook,
            scorer,
            cfg_,
            dataset,
        )
        best_loss = 1e9
        best_loss = pre_trainer.train()

        # manually release memory to prevent oom.
        del dataset
        del model
        del scorer
        del codebook
        del pre_trainer
        gc.collect()

    if cfg.data.elem_order == "optimized" and cfg.trainer.run_mode in [
        "train",
        "full",
        "eval",
    ]:
        # loop over order_pairs
        if cfg.data.order_pretrain_flag:
            order_dir = "./order-pretrain"
            if cfg.data.order_selected is not None:
                selected = cfg.data.order_selected
                orders = [f"order_pairs_noinjected_{i}.npy" for i in selected]
            else:
                orders = [
                    f for f in os.listdir(order_dir) if f.startswith("order_pairs")
                ]
        else:
            if cfg.data.order_pretrain_path is not None:
                order_dir = cfg.data.order_pretrain_path
                if cfg.data.order_selected is not None:
                    selected = cfg.data.order_selected
                    orders = [
                        os.path.join(order_dir, f"order_pairs_noinjected_{i}.npy")
                        for i in selected
                    ]
                else:
                    orders = [
                        f for f in os.listdir(order_dir) if f.startswith("order_pairs")
                    ]
            else:
                order_dir = os.path.join(
                    ROOT,
                    "./order-pretrain",
                )
                selected = [48]  # [0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30], different epochs
                orders = [
                    os.path.join(order_dir, f"order_pairs_noinjected_{i}.npy")
                    for i in selected
                ]
        for order in orders:
            cfg.data.extern_order_map = os.path.join(order_dir, order)
            cfg.data.order_id = order.split("_")[-1].split(".")[0]
            dataset = instantiate(cfg.dataset)(
                dataset_cfg=cfg.data,
            )
            id = order.split("_")[-1].split(".")[0]
            logger.info(f"Current order map id: {id}")
            cfg.trainer.log_dir = f"./order_{id}"
            cfg.trainer.ckpt_dir = f"./order_{id}/checkpoints"

            # EVAL MODE
            if cfg.trainer.run_mode in ["eval"]:
                if cfg.data.order_sweep_dir is not None:
                    sweep_dir = cfg.data.order_sweep_dir
                else:
                    sweep_dir = os.path.join(
                        ROOT,
                        "./multirun/.", # should manually set run_dir
                    )
                cfg.trainer.codebook_dir = os.path.join(
                    sweep_dir, f"order_{id}/checkpoints/last/codebook.pth"
                )
                cfg.trainer.model_dir = os.path.join(
                    sweep_dir, f"order_{id}/checkpoints/last/model.pth"
                )
                cfg.trainer.ckpt_dir = os.path.join(
                    sweep_dir, f"order_{id}/checkpoints"
                )

            # TRAINING PROCESS
            scorer = Scorer(cfg.scorer, dataset.vocab_size, cfg.data.mix_visual_feat)
            codebook = Codebook(dataset.vocab_size, cfg.model.n_embd, dataset.pad_token)
            model = instantiate(cfg.model)(
                vocab_size=dataset.vocab_size, block_size=dataset.max_seq_length
            )
            trainer = Trainer(
                model,
                codebook,
                scorer,
                cfg,
                dataset,
            )
            best_loss = 1e9
            if cfg.trainer.run_mode == "splitfid":
                # compute the fid score for split samples
                trainer.datasplit_fid()
                return 0
            if cfg.trainer.run_mode in ["train", "full"]:
                best_loss = trainer.train()
            if cfg.trainer.run_mode in ["eval", "full"]:
                trainer.eval()
            # manually release memory to prevent oom.
            trainer.tfwriter.close()
            trainer.wandbwriter.finish()
            del dataset
            del model
            del scorer
            del codebook
            del trainer
            gc.collect()

    else:
        dataset = instantiate(cfg.dataset)(
            dataset_cfg=cfg.data,
        )
        # TRAINING PROCESS
        scorer = Scorer(cfg.scorer, dataset.vocab_size, cfg.data.mix_visual_feat)
        codebook = Codebook(dataset.vocab_size, cfg.model.n_embd, dataset.pad_token)
        model = instantiate(cfg.model)(
            vocab_size=dataset.vocab_size, block_size=dataset.max_seq_length
        )   
        trainer = Trainer(
            model,
            codebook,
            scorer,
            cfg,
            dataset,
        )
        best_loss = 1e9
        if cfg.trainer.run_mode == "splitfid":
            # compute the fid score for split samples
            trainer.datasplit_fid()
            return 0
        if cfg.trainer.run_mode in ["train", "full"]:
            best_loss = trainer.train()
        if cfg.trainer.run_mode in ["eval", "full"]:
            trainer.eval()
        # manually release memory to prevent oom.
        trainer.tfwriter.close()
        trainer.wandbwriter.finish()
        del dataset
        del model
        del scorer
        del codebook
        del trainer
        gc.collect()

    # for OptunaSweeper
    return best_loss


if __name__ == "__main__":
    main()
