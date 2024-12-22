from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.k = cfg["model"]["codebook"]["k"]
        self.dim = cfg["model"]["codebook"]["dim"]
        self.codebook_dim = self.dim // 2
        self.beta = cfg["model"]["codebook"]["commitment_coefficient"]
        self.pre_quant_layer = torch.nn.Conv2d(self.dim, self.dim // 2, kernel_size=1)
        self.post_quant_layer = torch.nn.Conv2d(self.dim // 2, self.dim, kernel_size=1)

        self.register_buffer("lookup_table", torch.randn(self.k, self.codebook_dim))
        torch.nn.init.xavier_normal_(self.lookup_table)

        # EMA
        self.register_buffer("cluster_size", torch.zeros(self.k))
        self.enable_ema_update = False
        self.decay = 0.99  # EMA decay rate
        self.epsilon = 1e-5  # Small constant to prevent divide-by-zero errors

    def update_codebook(self, x, indices):
        x = x.permute(0, 2, 3, 1).contiguous()

        b, w, h, d = x.shape
        x = x.view(-1, d)
        indices = indices.view(-1)

        one_hot_assignments = torch.nn.functional.one_hot(indices, self.k).float()

        new_cluster_size = one_hot_assignments.sum(dim=0)
        self.cluster_size = (
            self.decay * self.cluster_size + (1 - self.decay) * new_cluster_size
        )

        weighted_sums = one_hot_assignments.T @ x  # Shape: (num_codebook_vectors, d)
        self.lookup_table = (
            self.decay * self.lookup_table + (1 - self.decay) * weighted_sums
        )

        n = self.cluster_size.sum()
        cluster_size_norm = (self.cluster_size + self.epsilon) / (n + self.epsilon) * n
        self.lookup_table = self.lookup_table / cluster_size_norm.unsqueeze(1).clamp(
            min=self.epsilon
        )

    def codebook_loss(
        self, x: torch.Tensor, x_e: torch.Tensor, q_x: torch.Tensor
    ) -> torch.Tensor:
        codebook_loss = 0
        if self.enable_ema_update:
            self.update_codebook(x, q_x)
        else:
            codebook_loss = torch.mean((x.detach() - x_e) ** 2)
        commitment_loss = self.beta * torch.mean((x - x_e.detach()) ** 2)
        return (1 - self.beta) * codebook_loss + commitment_loss

    @torch.no_grad()
    def decode(self, q_x, x=None):
        b, w, h = q_x.shape
        q_x = q_x.view(b, -1)
        x_e = (
            self.lookup_table[q_x]
            .view(b, h, w, self.codebook_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        if x is not None:
            x = x + (x - x_e)
        else:
            x = x_e
        x = self.post_quant_layer(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, d, h, w = x.shape
        x = self.pre_quant_layer(x)

        _x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.codebook_dim)

        distances = torch.cdist(_x, self.lookup_table)
        q_x = torch.argmin(distances, dim=1)

        x_e = (
            self.lookup_table[q_x]
            .view(b, h, w, self.codebook_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        codebook_loss = self.codebook_loss(x, x_e, q_x)

        # skip the gradiant from the codebook
        x = x + (x_e - x).detach()
        x = self.post_quant_layer(x)

        q_x = q_x.view(b, h, w)

        # Entropy loss (optional)
        e_mean = (
            F.one_hot(q_x.view(b, -1), num_classes=self.k)
            .view(-1, self.k)
            .float()
            .mean(0)
        )
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        config = {"loss": codebook_loss, "q": q_x, "perplexity": perplexity}
        return x, config
