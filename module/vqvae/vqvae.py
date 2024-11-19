import torch
from module.vqvae.encoder import ImageEncoder
from module.vqvae.codebook import Codebook
from module.vqvae.decoder import Decoder
from module.loss import VGGPerceptualLoss


class VQVAE(torch.nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg)
        self.codebook = Codebook(cfg)
        self.decoder = Decoder(cfg)
        self.act = torch.nn.Tanh()
        if self.cfg["model"]["enable_perceptual"]:
            self.perp_loss_fn = VGGPerceptualLoss()

    def reconstruction_loss(self, x, x_p) -> torch.Tensor:
        return torch.nn.functional.mse_loss(x, x_p)

    def generate(self, x):
        x = self.encoder(x)
        x, q_x, _ = self.codebook(x)
        x = self.decoder(x)
        x = self.act(x)
        return x, q_x

    def forward(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
            x: (B, C, H, W)
            label: (B) or (B, N)

        Return:
            x_h: (B,C,H,W), reconstructed image
            (codebook loss, recoonstructed loss)
        """
        perp_loss = 0
        z_h = self.encoder(x)
        z_h, discrete_h, codebook_loss = self.codebook(z_h)
        x_h = self.act(self.decoder(z_h))
        x_h = self.decoder(z_h)
        reco_loss = self.reconstruction_loss(x, x_h)

        if self.cfg["model"]["enable_perceptual"]:
            perp_loss = self.perp_loss_fn(
                x_h, x, feature_layers=[0, 1, 2, 3], style_layers=[]
            )
        reco_loss += perp_loss
        return x_h, (codebook_loss, reco_loss)
