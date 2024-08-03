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

from module.vae import AutoEncoder, VariationalAutoEncoder
from module.data import get_dataloader
from module.loss import ReconstructionLoss
from module.constant import ModelEnum
from helper import AverageMeter, export_model
from scripts.generate import ModelHelper

####################
WANDB_ENABLE = False
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("output.log")],
)
####################


def custom_print(message):
    print(message)  # Print to notebook cell
    logging.info(message)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", required=True)
    parser.add_argument("-d", "--device", dest="device", default="cpu")
    parser.add_argument("-p", "--data-path", dest="data_path")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("-n", "--num-worker", dest="num_workers", type=int)
    parser.add_argument("-e", "--epochs", dest="epochs", type=int)
    parser.add_argument("-w", "--wandb-key", dest="wandb_key", type=str)
    parser.add_argument("--debug", dest="debug", default=False, type=bool)

    args = parser.parse_args()
    return args


def per_epoch(
    cfg: Dict,
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
        img, label = batch[0].to(device), batch[1].to(device)
        pred, kl_loss = model(img, label)
        if train:
            _loss: torch.Tensor = loss_fn(pred, img)
            if cfg['model']['enable_kl_loss'] and kl_loss:
                _loss = (1 - cfg['train']['kl_loss_weight']) * _loss + cfg['train']['kl_loss_weight'] * kl_loss
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
        else:
            # eval
            with torch.no_grad():
                _loss: torch.Tensor = loss_fn(pred, img)
                if cfg['model']['enable_kl_loss'] and kl_loss:
                    _loss = (1 - cfg['train']['kl_loss_weight']) * _loss + cfg['train']['kl_loss_weight'] * kl_loss

        with torch.no_grad():
            # compute metrics
            _loss = _loss.detach().cpu().item()
            loss.update(_loss)

        if not WANDB_ENABLE:
            iters.set_description(f"[{mode}] loss: {_loss: .3f} avgLoss: {loss.avg}")
        else:
            if idx % 50 == 0:
                custom_print(
                    f"[{mode}] step: {idx}, loss: {_loss: .3f}, avgLoss: {loss.avg}"
                )
                if WANDB_ENABLE:
                    wandb.log({f"{mode}-step-loss": loss.avg})
    if WANDB_ENABLE:
        wandb.log({f"{mode}-loss": loss.avg})
    return loss.avg


def execute(cfg: Dict, device: str, debug=False):
    # load model

    if ModelEnum.AUTO_ENCODER.value == cfg["model"]["name"]:
        model = AutoEncoder(cfg)
    elif ModelEnum.VARIATIONAL_AUTO_ENCODER.value == cfg["model"]["name"]:
        model = VariationalAutoEncoder(cfg)
    else:
        raise ValueError(
            f"Model {cfg['model']['name']} not supported. Available models are {ModelEnum.list()}"
        )
    model = model.to(device)

    # load dataloader
    train_dl, val_dl = get_dataloader(cfg, testing_enable = debug)

    # define loss_fn and optimizer
    optim_clz: torch.optim.Optimizer = getattr(torch.optim, cfg["train"]["optim"])
    optimizer = optim_clz(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_dl) * cfg["train"]["epochs"],
        eta_min=cfg["train"]["scheduler"]["eta_min"],
    )

    loss_fn = ReconstructionLoss(cfg)
    GLOBAL_VAL_LOSS = 1e4
    train_loss, val_loss = 0, 0
    # execute pipeline
    for epoch in range(cfg["train"]["epochs"]):
        custom_print(f"Epoch: {epoch+1}")
        custom_print("Start Training Step")
        train_loss = per_epoch(cfg, model, train_dl, optimizer, loss_fn, device, True)
        logging.info("Start Validating Step")
        val_loss = per_epoch(cfg, model, val_dl, optimizer, loss_fn, device, False)
        scheduler.step()

        # TODO: Save checkpoint
        if GLOBAL_VAL_LOSS > val_loss:
            GLOBAL_VAL_LOSS = val_loss
            export_model(model, cfg=cfg, without_wandb=not WANDB_ENABLE)
            if cfg["generate_samples"]:
                model_helper = ModelHelper(
                    cfg=cfg, model_chkpt_path=cfg["model"]["export_path"], device="cpu"
                )
                model_helper.refresh_weights(model)
                model_helper.generate_and_save(n_items=4, file_prefix=epoch)

        logging.info(
            f"Epoch Summary: [{epoch+1}] train_loss: {train_loss: .5f}, val_loss: {val_loss: .5f}, lr: {scheduler.get_last_lr()}, GLOBAL_VAL_LOSS: {GLOBAL_VAL_LOSS: .5f}"
        )


if __name__ == "__main__":
    args = get_parser()
    custom_print("Start Training")
    config_path = os.path.join("config", f"{args.config}.yaml")
    custom_print(f"CONFIG: {config_path}")
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

    execute(cfg, args.device, debug=args.debug)

    if WANDB_ENABLE:
        wandb.finish()
    custom_print("Training Completed!")
