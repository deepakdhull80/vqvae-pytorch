import torch
from module.vqvae.encoder import ImageEncoder
from module.vqvae.codebook import Codebook
from module.vqvae.decoder import Decoder


class VQVAE(torch.nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg)
        self.codebook = Codebook(cfg)
        self.decoder = Decoder(cfg)

    def reconstruction_loss(self, x, x_p) -> torch.Tensor:
        return torch.nn.functional.mse_loss(x, x_p)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
            x: (B, C, H, W)
            label: (B) or (B, N)

        Return:
            x_h: (B,C,H,W), reconstructed image
            (codebook loss, recoonstructed loss)
        """
        z_h = self.encoder(x)
        z_h, discrete_h, codebook_loss = self.codebook(z_h)
        x_h = self.decoder(z_h)
        reco_loss = self.reconstruction_loss(x, x_h)
        return x_h, (codebook_loss, reco_loss)
