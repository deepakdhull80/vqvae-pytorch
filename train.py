import logging
import os
from typing import Dict, Tuple
import argparse
import torch.utils
import torch.utils.data
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from module.vae import AutoEncoder
from module.data import get_dataloader
from module.loss import ReconstructionLoss
from helper import AverageMeter, export_model

####################
WANDB_ENABLE = False
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("output.log"),
                        logging.StreamHandler()
                    ])
####################


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", required=True)
    parser.add_argument("-d", "--device", dest="device", default="cpu")
    parser.add_argument("-p", "--data-path", dest="data_path")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("-n", "--num-worker", dest="num_workers", type=int)
    parser.add_argument("-e", "--epochs", dest="epochs", type=int)
    parser.add_argument("-w", "--wandb-key", dest="wandb_key", type=str)

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
    if not WANDB_ENABLE:
        iters = tqdm(enumerate(dl), total=len(dl))
    else:
        iters = enumerate(dl)
    
    model = model.train() if train else model.eval()
    loss = AverageMeter()

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
            _loss = _loss.detach().cpu().item()
            loss.update(_loss)

        if not WANDB_ENABLE:
            iters.set_description(f"[{mode}] loss: {_loss: .3f} avgLoss: {loss.avg}")
        else:
            if idx % 20 == 0:
                logging.info(f"[{mode}] step: {idx}, loss: {_loss: .3f}, avgLoss: {loss.avg}")
    if WANDB_ENABLE:
        wandb.log({f"{mode}-loss": loss.avg})
    return loss.avg


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
    GLOBAL_VAL_LOSS = 1e4
    # execute pipeline
    for epoch in range(cfg["train"]["epochs"]):
        logging.info(f"Epoch: {epoch+1}")
        logging.info("Start Training Step")
        train_loss = per_epoch(model, train_dl, optimizer, loss_fn, device, True)
        logging.info("Start Validating Step")
        val_loss = per_epoch(model, val_dl, optimizer, loss_fn, device, False)
        scheduler.step()

        # TODO: Save checkpoint
        if GLOBAL_VAL_LOSS > val_loss:
            GLOBAL_VAL_LOSS = val_loss
            export_model(model, cfg=cfg)

        logging.info(
            f"Epoch Summary: [{epoch+1}] train_loss: {train_loss: .5f}, val_loss: {val_loss: .5f}, lr: {scheduler.get_last_lr()}, GLOBAL_VAL_LOSS: {GLOBAL_VAL_LOSS: .5f}"
        )


if __name__ == "__main__":
    args = get_parser()
    logging.info("Start Training")
    config_path = os.path.join("config", f"{args.config}.yaml")
    logging.info(f"CONFIG: {config_path}")
    assert os.path.exists(config_path), "Config file not found: {}".format(config_path)
    cfg = yaml.safe_load(open(config_path, "r"))

    cfg["data"]["data_prefix_path"] = (
        args.data_path if args.data_path else cfg["data"]["data_prefix_path"]
    )
    assert os.path.exists(
        cfg["data"]["data_prefix_path"]
    ), "Config file not found: {}".format(cfg["data"]["data_prefix_data"])

    cfg["data"]["batch_size"] = (
        args.batch_size if args.batch_size else cfg["data"]["batch_size"]
    )
    cfg["data"]["num_workers"] = (
        args.num_workers if args.num_workers else cfg["data"]["num_workers"]
    )
    cfg["data"]["epochs"] = args.epochs if args.epochs else cfg["train"]["epochs"]

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        WANDB_ENABLE = True

    if WANDB_ENABLE:
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"{cfg['model']['name']}-project",
            # track hyperparameters and run metadata
            config=cfg,
        )

    execute(cfg, args.device)

    if WANDB_ENABLE:
        wandb.finish()
    logging.info("Training Completed!")
