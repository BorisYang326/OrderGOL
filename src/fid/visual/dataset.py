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
from tool import configure_logging


### Config Logger
configure_logging()
logger = logging.getLogger()

class CustomCrelloDataset(Dataset):
    def __init__(self, hf_dataset: hfdatasets.Dataset, transforms: transforms.Compose, mode: str = 'mae-pretrain', cache_dir: str = './cache', split: str = 'train'):
        assert mode in ['mae-pretrain', 'mae-finetune', 'detection-finetune']
        assert split in ['train', 'val']
        self.hfdataset = hf_dataset
        self.mode = mode
        self.split = split
        self.transform = transforms
        self.cache_dir = cache_dir
        self.image_cache = {}
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{mode}_{split}_image_cache.pkl")
        
        self._read_cache()

    def _read_cache(self):
        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached images from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.image_cache = pickle.load(f)
        else:
            logger.info("Cache not found. Creating new cache...")
            self._cache_images()
            self._write_cache()

    def _cache_images(self):
        for idx in tqdm(range(len(self.hfdataset)), desc=f"Caching {self.split} images for {self.mode}"):
            sample = self.hfdataset[idx]
            image = sample['preview'].convert('RGB')
            # transformed_image = self.transform(image)
            self.image_cache[idx] = image

    def _write_cache(self):
        logger.info(f"Saving cache to {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.image_cache, f)

    def __len__(self):
        return len(self.hfdataset)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        raw_image = self.image_cache[idx]
        transformed_image = self.transform(raw_image)
        if self.mode in ['mae-pretrain', 'mae-finetune']:
            # Get the transformed image from cache
            return transformed_image
        elif self.mode == 'detection-finetune':
            sample = self.hfdataset[idx]
            # Process bounding boxes
            bboxes = []
            for t, l, w, h in zip(sample['top'], sample['left'], sample['width'], sample['height']):
                bboxes.append([l, t, l+w, t+h])  # [x1, y1, x2, y2] format
            
            # Convert to tensor and normalize
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            bboxes = bboxes.clamp(0, 1)  # Ensure values are between 0 and 1

            return transformed_image, bboxes
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _filter_by_len(self, max_elements: int):
        """Return a list of indices where the sample length is less than or equal to max_length."""
        valid_indices = []
        for i, example in tqdm(
            enumerate(self.hfdataset),
            total=len(self.hfdataset),
            desc=f"Filtering dataset by length <{max_elements}>",
        ):
            if example['length'] <= max_elements:
                valid_indices.append(i)
        return valid_indices


def set_dataloader(args, hf_dataset: hfdatasets.Dataset, transforms: Tuple[transforms.Compose, transforms.Compose]):
    if args.max_elements is not None:
        valid_indices = CustomCrelloDataset(hf_dataset, transforms, args.mode, args.cache_dir, 'train')._filter_by_len(args.max_elements)
    else:
        valid_indices = list(range(len(hf_dataset)))

    ### Split dataset
    len_samples = len(valid_indices)
    train_indices, val_indices = random_split(
        valid_indices,
        [int(len_samples * args.split_ratio), len_samples - int(len_samples * args.split_ratio)],
        torch.Generator().manual_seed(args.seed),
    )

    # Create subsets of the original dataset
    train_hf_dataset = hf_dataset.select(train_indices)
    val_hf_dataset = hf_dataset.select(val_indices)
    train_transform, val_transform = transforms
    # Create separate CustomCrelloDataset instances for train and val
    train_dataset = CustomCrelloDataset(train_hf_dataset,train_transform, args.mode, args.cache_dir, 'train')
    val_dataset = CustomCrelloDataset(val_hf_dataset, val_transform, args.mode, args.cache_dir, 'val')

    logger.info(f"Train dataset loaded with {len(train_dataset)} samples")
    logger.info(f"Test dataset loaded with {len(val_dataset)} samples")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader