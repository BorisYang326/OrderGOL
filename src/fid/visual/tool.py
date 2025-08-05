import logging
import datetime
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader, random_split
from typing import Union, Tuple
import datasets as hfdatasets
import os
import pickle
import math

def configure_logging(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)  # Set the default logger level to debug

    # Define a custom filter to include microseconds in the log record
    class MicrosecondFilter(logging.Filter):
        def filter(self, record):
            record.customasctime = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S,%f"
            )
            return True

    # Create a formatter that uses our custom timestamp
    formatter = logging.Formatter(
        "[%(customasctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    # Clear existing handlers to prevent duplicate logs
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Configure a new stream handler
    chandler = logging.StreamHandler()
    chandler.setLevel(level)
    chandler.setFormatter(formatter)
    chandler.addFilter(MicrosecondFilter())

    # Add the new handler to the logger
    logger.addHandler(chandler)

### Config Logger
configure_logging()
logger = logging.getLogger()






def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=None,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")

def get_mae_transforms(args):
    if args.mode == "mae-pretrain":
        logger.info("Config MAE <Pretrain> Transform")
        # We do not use normalization for pretraining since the statistics of crello is not the same as imagenet
        if args.crello_no_aug:
            transform_train = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)), 
                _convert_image_to_rgb,
                transforms.ToTensor(),
            ])
            transform_val = transform_train
        else:
            transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
            transform_val = transforms.Compose([
                    transforms.Resize(args.input_size, interpolation=3),
                    transforms.CenterCrop(args.input_size),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    else:
        logger.info("Config MAE <Finetune> Transform")
        transform_train = build_transform(is_train=True, args=args)
        transform_val = build_transform(is_train=False, args=args)
    return transform_train, transform_val

def load_arrow_dataset(datafile_raw_path: str) -> hfdatasets.Dataset:
    all_datasets = []
    # Assume each batch folder starts with 'batch_'
    batch_folders = [
        f for f in os.listdir(datafile_raw_path) if f.startswith("batch_")
    ]
    batch_folders.sort()  # Ensure correct order

    for batch_folder in batch_folders:
        dataset_path = os.path.join(datafile_raw_path, batch_folder)
        dataset_ = hfdatasets.load_from_disk(dataset_path)
        all_datasets.append(dataset_)

    # Use concatenate_datasets to merge all datasets into one
    dataset = hfdatasets.concatenate_datasets(all_datasets)
    return dataset