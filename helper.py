from typing import Dict
import torch
import wandb


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def export_model(model: torch.nn.Module, cfg: Dict):
    torch.save(model.state_dict(), cfg["model"]["export_path"])
    artifact = wandb.Artifact(cfg["model"]["name"], type="model")
    artifact.add_file(cfg["model"]["export_path"])

    # Log the artifact to WandB
    wandb.log_artifact(artifact)

    print("Model checkpoint saved!")
