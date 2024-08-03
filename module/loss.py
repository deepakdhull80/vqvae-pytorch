import torch
import torch.nn.functional as F


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg['train']['loss_fn'] == 'smoothL1':
            self.loss_fn = torch.nn.SmoothL1Loss()
        elif cfg['train']['loss_fn'] == 'mae':
            self.loss_fn = torch.nn.L1Loss()
        elif cfg['train']['loss_fn'] == 'mse':
            self.loss_fn = F.mse_loss
        elif cfg['train']['loss_fn'] == 'bce':
            self.loss_fn = torch.nn.BCELoss()
        else:
            raise ValueError(f"loss_fn {cfg['train']['loss_fn']} not found")

    def forward(self, predict, actual) -> torch.Tensor:
        return self.loss_fn(predict, actual)
