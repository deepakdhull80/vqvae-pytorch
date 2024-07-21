import os
import random
from typing import Dict, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
import torchvision
from torchvision.transforms import (
    Compose,
    RandomAdjustSharpness,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    RandomAutocontrast,
)


class RandomDataset(Dataset):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __getitem__(self, index):
        return torch.randn(3, 256, 256)

    def __len__(self):
        return self.n_samples


class Transformer(object):
    def __init__(self, cfg: Dict) -> None:
        self.augments = Compose(
            [
                Resize(size=(cfg["img_shape"], cfg["img_shape"]), antialias=True),
                RandomHorizontalFlip(p=cfg["data"]["augmentation_probability"]),
                RandomVerticalFlip(p=cfg["data"]["augmentation_probability"]),
                RandomAdjustSharpness(2, p=cfg["data"]["augmentation_probability"]),
                RandomAutocontrast(p=cfg["data"]["augmentation_probability"]),
            ]
        )

    def __call__(self, x: torch.Tensor):
        return self.augments(x)


class COCODataset(Dataset):
    def __init__(self, cfg: Dict, image_list: List) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_prefix_path = cfg["data"]["data_prefix_path"]
        self.image_list = image_list
        self.transform = Transformer(cfg)

    def normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        return (img / 255.0 - 0.5) / 0.5

    def correct_img_channels(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image

    def __getitem__(self, index):
        file = self.image_list[index]
        path = f"{self.data_prefix_path}/{file}"
        img = torchvision.io.read_image(path)
        img = self.correct_img_channels(img)
        img = self.transform(img)
        img = img.float()
        img = self.normalize_img(img)
        return img

    def __len__(self):
        return len(self.image_list)


def custom_collate_fn(batch: torch.Tensor):
    return default_collate(batch)


def get_dataloader(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    random.seed(cfg["random_seed"])

    # These comments are for debugging purposes
    # train_ds = RandomDataset(1000)
    # val_ds = RandomDataset(100)

    imgs = os.listdir(cfg["data"]["data_prefix_path"])
    random.shuffle(imgs)

    train_size = int(len(imgs) * cfg["data"]["train_size"])
    train_image_li = imgs[:train_size]
    val_image_li = imgs[train_size:]

    train_ds = COCODataset(cfg=cfg, image_list=train_image_li)
    val_ds = COCODataset(cfg=cfg, image_list=val_image_li)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        collate_fn=custom_collate_fn,
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        collate_fn=custom_collate_fn,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    return train_dl, val_dl
