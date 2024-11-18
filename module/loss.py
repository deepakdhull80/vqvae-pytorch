import torch
import torch.nn.functional as F


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg["train"]["loss_fn"] == "smoothL1":
            self.loss_fn = torch.nn.SmoothL1Loss()
        elif cfg["train"]["loss_fn"] == "mae":
            self.loss_fn = torch.nn.L1Loss()
        elif cfg["train"]["loss_fn"] == "mse":
            self.loss_fn = F.mse_loss
        elif cfg["train"]["loss_fn"] == "bce":
            self.loss_fn = torch.nn.BCELoss()
        elif cfg["train"]["loss_fn"] == "gaussian_likelihood":
            self.loss_fn = self.gaussian_likelihood
            self.logscale = torch.nn.Parameter(torch.Tensor([0.0]))
        else:
            raise ValueError(f"loss_fn {cfg['train']['loss_fn']} not found")

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(self.logscale).to(x.device)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def forward(self, predict, actual) -> torch.Tensor:
        return self.loss_fn(predict, actual)
