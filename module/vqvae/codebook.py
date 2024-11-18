from typing import Tuple
import torch


class Codebook(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.k = cfg["model"]["codebook"]["k"]
        self.dim = cfg["model"]["codebook"]["dim"]
        self.beta = cfg["model"]["codebook"]["commitment_coefficient"]
        self.lookup_table = torch.nn.parameter.Parameter(
            data=torch.randn(self.k, self.dim)
        )

    def codebook_loss(self, x: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        codebook_loss = torch.nn.functional.mse_loss(x.detach(), x_e)
        commitment_loss = torch.nn.functional.mse_loss(x, x_e.detach())
        loss = codebook_loss + self.beta * commitment_loss
        return loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, d, h, w = x.shape

        _x = x.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        distances = torch.cdist(_x, self.lookup_table)
        q_x = torch.argmin(distances, dim=1)

        x_e = self.lookup_table[q_x].view(b, h, w, d).permute(0, 3, 1, 2)
        q_x = q_x.view(b, h, w)

        codebook_loss = self.codebook_loss(x, x_e)
        return x_e, q_x, codebook_loss
