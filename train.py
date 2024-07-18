import os
from typing import Dict, Tuple
import argparse
import torch.utils
import torch.utils.data
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from module.vae import AutoEncoder
from module.data import get_dataloader
from module.loss import ReconstructionLoss


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", required=True)
    parser.add_argument("-d", "--device", dest="device", default="cpu")

    args = parser.parse_args()
    return args


def per_epoch(
    model: nn.Module,
    dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    train: bool = True,
):
    iters = tqdm(enumerate(dl), total=len(dl))
    model = model.train() if train else model.eval()
    loss = 0
    mode = "Train" if train else "Eval"
    for idx, batch in iters:
        img = batch.to(device)
        pred = model(img)
        if train:
            _loss: torch.Tensor = loss_fn(pred, img)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _loss: torch.Tensor = loss_fn(pred, img)

        with torch.no_grad():
            # compute metrics
            loss += _loss.detach().cpu().item()

        iters.set_description(
            f"[{mode}] loss: {_loss.detach().cpu().item(): .3f} avgLoss: {loss/(idx+1)}"
        )
    return loss / len(dl)


def execute(cfg: Dict, device: str):
    # load model
    model = AutoEncoder(cfg)
    model = model.to(device)

    # load dataloader
    train_dl, val_dl = get_dataloader(cfg)

    # define loss_fn and optimizer
    optim_clz: torch.optim.Optimizer = getattr(torch.optim, cfg["train"]["optim"])
    optimizer = optim_clz(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_dl) * cfg["train"]["epochs"],
        eta_min=cfg["train"]["scheduler"]["eta_min"],
    )

    loss_fn = ReconstructionLoss()

    # execute pipeline
    for epoch in range(cfg["train"]["epochs"]):
        print(f"Epoch: {epoch+1}")
        print("Start Training Step")
        train_loss = per_epoch(model, train_dl, optimizer, loss_fn, device, True)
        print("Start Validating Step")
        val_loss = per_epoch(model, val_dl, optimizer, loss_fn, device, False)
        scheduler.step(epoch + 1)

        # TODO: Save checkpoint

        print(
            f"Epoch Summary: [{epoch+1}] train_loss: {train_loss: .3f}, val_loss: {val_loss: .3f}, lr: {scheduler.get_lr()}"
        )


if __name__ == "__main__":
    args = get_parser()
    print("Start Training")
    config_path = os.path.join("config", f"{args.config}.yaml")
    print(f"CONFIG: {config_path}")
    assert os.path.exists(config_path), "Config file not found: {}".format(config_path)
    cfg = yaml.safe_load(open(config_path, "r"))

    execute(cfg, args.device)
    print("Training Completed!")
