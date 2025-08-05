import os
import sys
import torch
from hydra.core.config_store import ConfigStore
from torch.utils.data.dataloader import DataLoader
from hydra.utils import instantiate
from hydra import initialize, compose
from typing import Any
import argparse
from tqdm import tqdm
import wandb
import omegaconf
from omegaconf import OmegaConf
from helpers import (
    set_input,
    tokenize_tok,
    get_weight_mask,
    split_dataset,
    reconstruct_and_save,
    extract_embeddings_and_save,
    retrieve_nearest_neighbors,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
from src.configs import DataConfig, ScorerConfig, TrainerConfig
from src.trainer import Trainer
from src.model.codebook import Codebook
from src.scorer import Scorer
from src.sampling import SAMPLING_CONFIG_DICT

# from src.fid.design_vae import DesignVAE
from src.fid.seq.play_design_vae import DesignVAE
from src.fid.seq.fidnet_design_ae import DesignAE

MODELS = {"vae": DesignVAE, "ae": DesignAE}


def train(dataset, model, optimizer, cfg, device, args, wandb_writer, loaders,weight_mask):
    train_loader, test_loader = loaders['train'],loaders['test']
    best_loss = float("inf")
    is_improved = False
    postfix_ = cfg.trainer.exp_name.split('_')[0]
    for epoch in range(cfg.trainer.max_epochs):
        total_train_loss = 0
        total_test_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        # train
        model.train()
        for it, (seq, patches, mask, canvas_, order_dics) in pbar:
            seq, patches, mask, canvas, order_dics = set_input(
                seq, patches, mask, canvas_, order_dics, device
            )
            if args.decoder_type == "at":
                enc_tok, enc_mask = tokenize_tok(dataset, seq, mask, canvas, False)
                dec_tok, dec_mask = tokenize_tok(
                    dataset,
                    seq,
                    mask,
                    canvas,
                    True if args.decoder_type == "at" else False,
                )
                src = enc_tok
                src_mask = enc_mask
                tgt = dec_tok[:, :-1]
                tgt_mask = dec_mask[:, :-1]
                y = dec_tok[:, 1:]
            else:
                enc_tok, enc_mask = tokenize_tok(dataset, seq, mask, canvas, False)
                src = enc_tok
                src_mask = enc_mask
                tgt = enc_tok
                tgt_mask = enc_mask
                y = enc_tok
            optimizer.zero_grad()
            loss = model.get_loss(src, src_mask, tgt, tgt_mask,wandb_writer,weight_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # total_train_loss += loss.item()
            pbar.set_description(
                f"Epoch {epoch} iter {it}: train loss {loss.item():.5f}"
            )
        # test
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                for it, (seq, patches, mask, canvas_, order_dics) in enumerate(test_loader):
                    seq, patches, mask, canvas, order_dics = set_input(
                        seq, patches, mask, canvas_, order_dics, device
                    )
                    if args.decoder_type == "at":
                        enc_tok, enc_mask = tokenize_tok(dataset, seq, mask, canvas, False)
                        dec_tok, dec_mask = tokenize_tok(
                            dataset,
                            seq,
                            mask,
                            canvas,
                            True if args.decoder_type == "at" else False,
                        )
                        src = enc_tok
                        src_mask = enc_mask
                        tgt = dec_tok[:, :-1]
                        tgt_mask = dec_mask[:, :-1]
                        y = dec_tok[:, 1:]
                    else:
                        enc_tok, enc_mask = tokenize_tok(dataset, seq, mask, canvas, False)
                        src = enc_tok
                        src_mask = enc_mask
                        tgt = enc_tok
                        tgt_mask = enc_mask
                        y = enc_tok
                    loss = model.get_loss(src, src_mask, tgt, tgt_mask,wandb_writer,weight_mask)
                    total_test_loss += loss.item()
            test_loss = total_test_loss / len(test_loader)
            print(f"Epoch {epoch}, Test Loss: {test_loss}")
            wandb_writer.log({"test_loss": test_loss})
        if test_loader is not None:
            if test_loss < best_loss:
                best_loss = test_loss
                is_improved = True
            else:
                is_improved = False
        else:
            if epoch == args.epochs - 1:
                # store the last model
                is_improved = True
        if is_improved:
            if args.model_name == "vae":
                torch.save(
                    model.state_dict(),
                    f"{args.out_dir}/{postfix_}_design_vae_w{int(args.visual_balance_factor)}_{model.decoder_type}.pth",
                )
                print(
                    f"Model saved at {args.out_dir}/{postfix_}_design_vae_w{int(args.visual_balance_factor)}_{model.decoder_type}.pth"
                )
            else:
                torch.save(
                    model.state_dict(),
                    f"{args.out_dir}/{postfix_}_design_ae_w{int(args.visual_balance_factor)}.pth",
                )
                print(f"Model saved at {args.out_dir}/{postfix_}_design_ae_w{int(args.visual_balance_factor)}.pth")


def validation(dataset, model: Any, val_loader, cfg, device, side_trainer, args):
    model.load_pretrained(os.path.dirname(args.out_dir), args.visual_balance_factor)
    ## validation
    curr_path = ".."
    sampling_cfg = OmegaConf.structured(SAMPLING_CONFIG_DICT[cfg.trainer.sampling])
    OmegaConf.set_struct(sampling_cfg, False)
    if "temperature" in cfg.trainer:
        sampling_cfg.temperature = cfg.trainer.temperature
    if "top_p" in cfg.trainer and sampling_cfg.name == "top_p":
        sampling_cfg.top_p = cfg.trainer.top_p
    side_trainer.sampling_cfg = sampling_cfg
    # Reconstruct and save using the validation set
    reconstruct_and_save(
        val_loader, model, device, side_trainer, dataset, sampling_cfg, curr_path
    )

    # # Extract embeddings and save using the validation set
    extract_embeddings_and_save(
        val_loader, model, device, side_trainer, dataset, curr_path
    )

    # Retrieve nearest neighbors based on the saved embeddings
    retrieve_nearest_neighbors(curr_path, side_trainer, 10, 10)


def main():
    # config inherited from MainConfig
    # init device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # load default config
    cs = ConfigStore.instance()
    cs.store(group="trainer", name="base_trainer_config", node=TrainerConfig)
    cs.store(group="data", name="base_data_config", node=DataConfig)
    cs.store(group="scorer", name="base_scorer_config", node=ScorerConfig)
    # global initialization for hydra
    initialize(version_base="1.2", config_path="../../../config")

    # dataset_cfg = OmegaConf.load("config/dataset/crello-1k.yaml")
    cfg = compose(
        config_name="main.yaml",
        overrides=[
            "+experiment=cgl-28k",
            # "+experiment=crello-10k-fine3k-pos256-light",
            "trainer.exp_name='cgl-28k_bs${data.batch_size}_epochs${trainer.max_epochs}'",
            "data.elem_order=random",
            "trainer.max_epochs=50",
            "data.shuffle_per_epoch=True",
            "data.use_cache=true",
            "data.batch_size=256",
            "trainer.wandb_mode=disabled",
        ],
        # overrides=["+experiment=crello-1k", "data.elem_order=raster"],
    )
    ## args for training VAE
    parser = argparse.ArgumentParser()
    # parser.add_argument("--out_dir", type=str, default="./src/fid/fid_weights")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_latent", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--kl_beta", type=float, default=1e-2)
    parser.add_argument("--visual_balance_factor", type=float, default=1.0)
    parser.add_argument("--is_visual_only", type=bool, default=False)
    parser.add_argument("--decoder_type", type=str, default="nat")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--valid_only", action="store_true")
    parser.add_argument("--wandb_mode", type=str, choices=['disabled','online','offline'])
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument(
        "--out_dir", type=str, default=f"{cfg.data.cache_root}/fid_weights"
    )
    parser.add_argument("--model_name", type=str, default="vae")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--full_train",action="store_true")
    parser.add_argument("--batch_size",type=int,default=256)
    args = parser.parse_args()
    cfg.trainer.max_epochs = args.epochs
    cfg.data.batch_size = args.batch_size
    print(f"Using batch size: {args.batch_size}")
    # init
    name_ = (
        f"design_vae_{args.decoder_type}_lr{args.lr}_beta{args.kl_beta}_{args.model_name}"
    )
    wandb_writer = wandb.init(
        project="Crello-Utilization", mode=args.wandb_mode, name=name_
    )
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=False
    )
    wandb.config.update(vars(args))
    # run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    if args.seed is not None:
        cfg.trainer.seed = cfg.seed = args.seed
    dataset = instantiate(cfg.dataset)(
        dataset_cfg=cfg.data,
    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_config_str = os.getenv("CUDA_VISIBLE_DEVICES")
    print(f"Using GPU {gpu_config_str}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_mask = get_weight_mask(dataset, device, args)
    if args.model_name == "vae":
        model = DesignVAE(
            dataset.vocab_size,
            dataset.pad_token,
            dataset.max_seq_length,
            args.d_model,
            args.nhead,
            args.num_encoder_layers,
            args.d_latent,
            decoder_type=args.decoder_type,
            beta=args.kl_beta,
            weight_mask=weight_mask,
        ).to(device)
    else:
        model = DesignAE(
            dataset.vocab_size, dataset.pad_token, dataset.max_seq_length
        ).to(device)

    # this part for init trainer for detokenized/rendering.
    scorer = Scorer(cfg.scorer, dataset.vocab_size, cfg.data.mix_visual_feat)
    codebook = Codebook(dataset.vocab_size, cfg.scorer.embed_size, dataset.pad_token)
    side_model = instantiate(cfg.model)(
        vocab_size=dataset.vocab_size, block_size=dataset.max_seq_length
    )
    side_trainer = Trainer(
        side_model,
        codebook,
        scorer,
        cfg,
        dataset,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.full_train:
        full_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        loaders = {'train':full_dataloader,'test':None,'val':None}
    else:
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        loaders = {'train':train_loader,'test':test_loader,'val':val_loader}
    if not args.valid_only:
        train(
            dataset,
            model,
            optimizer,
            cfg,
            device,
            args,
            wandb_writer,
            loaders,
            weight_mask,
        )
    if loaders['val'] is not None:
        validation(dataset, model, loaders['val'], cfg, device, side_trainer, args)


if __name__ == "__main__":
    main()
