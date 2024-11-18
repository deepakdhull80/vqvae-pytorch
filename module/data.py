import os
import random
import json
import importlib
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
import torchvision.transforms.functional as VF


class RandomDataset(Dataset):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __getitem__(self, index):
        return torch.randn(3, 256, 256)

    def __len__(self):
        return self.n_samples


class Transformer(object):
    def __init__(self, cfg: Dict, flag: bool = False) -> None:

        if not flag:
            self.augments = Compose(
                [
                    Resize(size=(cfg["img_shape"], cfg["img_shape"]), antialias=True),
                    RandomHorizontalFlip(p=cfg["data"]["augmentation_probability"]),
                    RandomVerticalFlip(p=cfg["data"]["augmentation_probability"]),
                    RandomAdjustSharpness(2, p=cfg["data"]["augmentation_probability"]),
                    RandomAutocontrast(p=cfg["data"]["augmentation_probability"]),
                ]
            )
        else:
            self.augments = Compose(
                [Resize(size=(cfg["img_shape"], cfg["img_shape"]), antialias=True)]
            )

    def __call__(self, x: torch.Tensor):
        return self.augments(x)


class COCODataset(Dataset):
    def __init__(
        self, cfg: Dict, image_list: List, disable_tansforms: bool = False
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_prefix_path = cfg["data"]["data_prefix_path"]
        self.image_list = image_list
        self.transform = Transformer(cfg, flag=disable_tansforms)

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
        return img, torch.tensor(1)

    def __len__(self):
        return len(self.image_list)


class COCOConditionalDataset(Dataset):
    def __init__(
        self, cfg: Dict, disable_tansforms: bool = False, train: bool = True
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_prefix_path = cfg["data"]["data_prefix_path"]
        self._mode = "train2017" if train else "val2017"
        self.image_path = os.path.join(self.data_prefix_path, self._mode)
        self.json_data = json.load(
            open(
                f"{self.data_prefix_path}/annotations/instances_{self._mode}.json", "r"
            )
        )
        self.annotations = self.filter_annotations(self.json_data["annotations"])
        self.transform = Transformer(cfg, flag=disable_tansforms)
        self.num_classes = self.cfg["data"]["num_classes"]
        self.normalization_type = cfg["data"]["normalization_type"]

    def filter_annotations(self, annotations: List):
        new_annotations = []
        for a in annotations:
            bbox = list(map(int, a["bbox"]))
            if (
                a["area"] >= self.cfg["data"]["filter"]["min_area"]
                and bbox[2] >= self.cfg["data"]["filter"]["min_width"]
                and bbox[3] >= self.cfg["data"]["filter"]["min_height"]
            ):
                # width and height should be greater than 1
                new_annotations.append(a)
        return new_annotations

    def normalize_img(
        self, img: torch.Tensor, normalization_type: str = "tanh"
    ) -> torch.Tensor:
        if normalization_type == "tanh":
            return (img / 255.0 - 0.5) / 0.5
        elif normalization_type == "sigmoid":
            return img / 255.0
        else:
            raise ValueError(f"Normalization type {normalization_type} not supported")

    def correct_img_channels(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image

    def __getitem__(self, index):
        annotation = self.annotations[index]

        path = f"{self.image_path}/{annotation['image_id']:012d}.jpg"
        img = torchvision.io.read_image(path)
        bbox = list(map(int, annotation["bbox"]))
        img = VF.crop(img, top=bbox[0], left=bbox[1], height=bbox[2], width=bbox[3])
        img = self.correct_img_channels(img)
        img = self.transform(img)
        img = img.float()
        img = self.normalize_img(img, normalization_type=self.normalization_type)
        y = torch.nn.functional.one_hot(
            torch.tensor(annotation["category_id"] - 1), num_classes=self.num_classes
        )
        return img, y

    def __len__(self):
        return len(self.annotations)


def custom_collate_fn(batch: torch.Tensor):
    return default_collate(batch)


def get_dataloader(
    cfg: Dict, testing_enable: bool = False
) -> Tuple[DataLoader, DataLoader]:
    random.seed(cfg["random_seed"])

    # These comments are for debugging purposes
    # train_ds = RandomDataset(1000)
    # val_ds = RandomDataset(100)

    imgs = os.listdir(cfg["data"]["data_prefix_path"])
    random.shuffle(imgs)

    train_size = int(len(imgs) * cfg["data"]["train_size"])
    train_image_li = imgs[:train_size]
    val_image_li = imgs[train_size:]

    if cfg["data"]["clz"] == "COCODataset":
        train_kwargs = {"cfg": cfg, "image_list": train_image_li}
        val_kwargs = {"cfg": cfg, "image_list": val_image_li}

    elif cfg["data"]["clz"] == "COCOConditionalDataset":
        train_kwargs = {
            "cfg": cfg,
            "disable_tansforms": False,
            "train": not testing_enable,
        }
        val_kwargs = {"cfg": cfg, "disable_tansforms": False, "train": False}
    else:
        raise ValueError(f"{cfg['data']['clz']} implementation not found.")

    module = importlib.import_module("module.data")
    clz = getattr(module, cfg["data"]["clz"])

    train_ds = clz(**train_kwargs)
    val_ds = clz(**val_kwargs)

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
