import os
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
import timm.optim.optim_factory as optim_factory
import logging
from tqdm import tqdm
import wandb
from timm.data.mixup import Mixup
from dataset import set_dataloader
from tool import (
    configure_logging,
    adjust_learning_rate,
    get_mae_transforms
)
import datasets as hfdatasets
from mae import MAE
from lightly.transforms import MAETransform
from lightly.models import utils
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

### Config Logger
configure_logging()
logger = logging.getLogger()


def train_per_epoch(model,dataloader,optimizer,criterion,epoch,args):
    total_loss = 0
    model.train()
    for steps,batch in tqdm(enumerate(dataloader),total=len(dataloader), desc=f"Epoch {epoch}", leave=False):
        lr_ = adjust_learning_rate(optimizer,steps / len(dataloader) + epoch,args)
        wandb.log({"lr": lr_})
        images = batch.to(DEVICE)  # views contains only a single view
        predictions, targets = model(images)
        loss = criterion(predictions, targets)
        total_loss += loss.detach()
        wandb.log({"loss": loss.detach()})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Train | Epoch: {epoch}, loss: {avg_loss:.5f}")
    return model,avg_loss


def validate_per_epoch(model, dataloader, criterion, epoch):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation | Epoch {epoch}", leave=False):
            images = batch.to(DEVICE)
            predictions, targets = model(images)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

            # Log decoded results to wandb
            if epoch % 10 == 0:  # Log every 10 epochs to avoid excessive logging
                # Decode the first 20 images in the batch
                original = images[:20].to(DEVICE)
                original, masked, reconstruction, reconstruction_visible = model.reconstruct(original)
                
                wandb.log({
                    "original": [wandb.Image(img.cpu()) for img in original],
                    "masked": [wandb.Image(img.cpu()) for img in masked],
                    "reconstruction": [wandb.Image(img.cpu()) for img in reconstruction],
                    "reconstruction_visible": [wandb.Image(img.cpu()) for img in reconstruction_visible],
                })

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Validation - epoch: {epoch}, epoch-avg-loss: {avg_loss:.5f}")
    wandb.log({"val_loss": avg_loss, "epoch": epoch})
    return avg_loss


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_workers", type=int, default=32)
    argparser.add_argument(
        "--dataset_path", type=str, default="/storage/dataset/crello/full_phres/raw"
    )
    argparser.add_argument("--cache_dir", type=str, default="/storage/dataset/crello/full_phres/cache")
    argparser.add_argument("--split_ratio", type=float, default=0.95)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--mode", type=str, default="mae-pretrain")
    argparser.add_argument("--batch_size", type=int, default=512)
    argparser.add_argument("--epochs", type=int, default=200)
    argparser.add_argument("--blr", type=float, default=1.5e-4)
    argparser.add_argument("--lr", type=float, default=None)
    argparser.add_argument("--input_size", type=int, default=224)
    argparser.add_argument("--weight_decay", type=float, default=0.05)
    argparser.add_argument("--max_elements", type=int, default=None)
    argparser.add_argument("--warmup_epochs", type=int, default=40)
    argparser.add_argument("--min_lr", type=float, default=1e-6)
    # Fintune params
    argparser.add_argument("--smoothing", type=float, default=0.1)
    argparser.add_argument("--mixup", type=float, default=0.8)
    argparser.add_argument("--cutmix", type=float, default=1.0)
    argparser.add_argument("--drop_path_rate", type=float, default=0.1)
    argparser.add_argument("--wandb_project", type=str, default="Crello-Utilization")
    argparser.add_argument("--wandb_mode", type=str, default="online")
    argparser.add_argument("--crello_no_aug", action="store_true")
    args = argparser.parse_args()
    ### wandb
    name_ = f"{args.mode}-bs{args.batch_size}-ep{args.epochs}"
    wandb.init(project=args.wandb_project, name=name_, mode=args.wandb_mode)
    wandb.config.update(args)
    assert args.mode in ["mae-pretrain", "mae-finetune","detection-finetune"]
    is_vit_pretrained = False if args.mode == "mae-pretrain" else True
    ### Load dataset
    try:
        hfdataset = hfdatasets.load_from_disk(args.dataset_path)
        logger.info(f"Dataset loaded with {len(hfdataset)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
    transforms = get_mae_transforms(args)
    train_dataloader, val_dataloader = set_dataloader(args, hfdataset, transforms)
    
    ### train model
    model = MAE(vit_pretrained=is_vit_pretrained,drp=args).to(DEVICE)
    ### set optimizer
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256
        logger.info(f"actual lr : {args.lr}")
    if is_vit_pretrained:
        # finetune
        param_groups = optim_factory.param_groups_layer_decay(model, args.weight_decay)
        optimizer = optim.AdamW(param_groups, lr=args.lr)
        if args.mode == "mae-finetune":
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        # mixup_fn = None
        # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        # if mixup_active:
        #     logger.info("Mixup is activated!")
        #     mixup_fn = Mixup(
        #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        #         label_smoothing=args.smoothing, num_classes=args.nb_classes)
    else:
        #mae-pretrain from scratch
        param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
        optimizer = optim.AdamW(param_groups, lr=args.lr,betas=(0.9,0.95))
        criterion = nn.MSELoss()
    ### train model
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model,_ = train_per_epoch(model,train_dataloader,optimizer,criterion,epoch,args)
        val_loss = validate_per_epoch(model,val_dataloader,criterion,epoch)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     model.save_model(os.path.join(os.path.dirname(__file__),"./weights/mae_pretrain.pth"))
        if epoch % 40 == 0:
            os.makedirs(os.path.join(os.path.dirname(__file__),f"runs/{name_}/weights/"),exist_ok=True)
            model.save_model(os.path.join(os.path.dirname(__file__),f"runs/{name_}/weights/epoch{epoch}.pth"))
if __name__ == "__main__":
    main()
