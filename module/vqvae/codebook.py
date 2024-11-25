from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.k = cfg["model"]["codebook"]["k"]
        self.dim = cfg["model"]["codebook"]["dim"]
        self.beta = cfg["model"]["codebook"]["commitment_coefficient"]

        self.register_buffer("lookup_table", torch.empty(self.k, self.dim))
        torch.nn.init.uniform_(self.lookup_table, -1 / self.k, 1 / self.k)

    @torch.no_grad()
    def decode(self, q_x):
        b, w, h = q_x.shape
        q_x = q_x.view(-1, 1)
        encodings = torch.zeros(q_x.shape[0], self.k, device=q_x.device)
        encodings.scatter_(1, q_x, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.lookup_table).view(b, w, h, -1)
        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.lookup_table**2, dim=1)
            - 2 * torch.matmul(flat_input, self.lookup_table.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.k, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.lookup_table).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        config = {
            "loss": loss,
            "q": encoding_indices.view(input_shape[:-1]),
            "perplexity": perplexity,
            "codebook_loss": q_latent_loss,
            "commitment_loss": e_latent_loss,
        }
        return quantized.permute(0, 3, 1, 2).contiguous(), config
