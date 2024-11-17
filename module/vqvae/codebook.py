from typing import Tuple
import torch

class Codebook(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.k = cfg['model']['codebook']['k']
        self.dim = cfg['model']['codebook']['dim']
        self.lookup_table = torch.nn.parameter.Parameter(data=torch.randn(1, 1, self.k, self.dim))
    
    def codebook_loss(self, x: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.mse_loss(x.detach(), x_e)
        return loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, d, h, w = x.shape
        x = x.view(b, d, h*w).permute(0, 2, 1).unsqueeze(2).contiguous()
        q_x = (
                torch
                .argmin(torch.linalg.norm(x-self.lookup_table, dim=-1), dim=-1)
                )
        x_e = (
            self.lookup_table[:, :, q_x]
            .squeeze(0)
            .squeeze(0)
            .permute(0, 2, 1)
            .view(b, d, h, w)
        )
        q_x = q_x.view(b, h, w)
        codebook_loss = self.codebook_loss(x.view(b, d, h, w), x_e)
        return x_e, q_x, codebook_loss