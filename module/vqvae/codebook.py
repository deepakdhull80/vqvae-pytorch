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
        self.lookup_table = torch.nn.parameter.Parameter(
            data=torch.randn(self.k, self.dim)
        )

        # EMA
        self.enable_ema_update = False
        self.decay = 0.99  # EMA decay rate
        self.epsilon = 1e-5  # Small constant to prevent divide-by-zero errors

    def update_codebook(self, x, indices):
        one_hot = torch.nn.functional.one_hot(indices, num_classes=self.k).type(x.dtype)
        count = one_hot.sum(dim=0)
        count = torch.max(count, torch.tensor(self.epsilon, device=x.device))

        updated_lookup = (one_hot.T @ x) / count.unsqueeze(1)
        self.lookup_table.data = (
            self.decay * self.lookup_table.data + (1 - self.decay) * updated_lookup
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
        return codebook_loss + commitment_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, d, h, w = x.shape

        _x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.dim)

        distances = torch.cdist(_x, self.lookup_table)
        q_x = torch.argmin(distances, dim=1)

        x_e = self.lookup_table[q_x].view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        codebook_loss = self.codebook_loss(x, x_e, q_x)

        x = x + (x - x_e).detach()

        q_x = q_x.view(b, h, w)
        return x, q_x, codebook_loss


# class Codebook(nn.Module):
#     def __init__(self, cfg):
#         super(Codebook, self).__init__()

#         self.cfg = cfg
#         self.k = cfg["model"]["codebook"]["k"]
#         self.dim = cfg["model"]["codebook"]["dim"]
#         self.beta = cfg["model"]["codebook"]["commitment_coefficient"]

#         self._embedding = torch.nn.Embedding(self.k, self.dim)
#         self._embedding.weight.data.uniform_(-1/self.k, 1/self.k)

#     def forward(self, inputs):
#         # convert inputs from BCHW -> BHWC
#         inputs = inputs.permute(0, 2, 3, 1).contiguous()
#         input_shape = inputs.shape

#         # Flatten input
#         flat_input = inputs.view(-1, self.dim)

#         # Calculate distances
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
#                     + torch.sum(self._embedding.weight**2, dim=1)
#                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

#         # Encoding
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], self.k, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)

#         # Quantize and unflatten
#         quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

#         # Loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())

#         loss = q_latent_loss + self.beta * e_latent_loss

#         quantized = inputs + (quantized - inputs).detach()
#         # avg_probs = torch.mean(encodings, dim=0)
#         # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

#         # convert quantized from BHWC -> BCHW
#         # return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
#         return quantized.permute(0, 3, 1, 2).contiguous(), encodings, loss
