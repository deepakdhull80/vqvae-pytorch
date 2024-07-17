from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, default_collate


class RandomDataset(Dataset):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __getitem__(self, index):
        return torch.randn(3, 256, 256)

    def __len__(self):
        return self.n_samples


def custom_collate_fn(batch: torch.Tensor):
    return default_collate(batch)


def get_dataloader(cfg: Dict) -> Tuple[DataLoader, DataLoader]:

    train_ds = RandomDataset(1000)
    val_ds = RandomDataset(100)

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
