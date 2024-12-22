import torch

# from vqtorch.nn import VectorQuant

from module.vqvae.encoder import ImageEncoder
from module.vqvae.codebook import Codebook
from module.vqvae.decoder import Decoder
from module.loss import VGGPerceptualLoss


class VQVAE(torch.nn.Module):
    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg)
        self.codebook = Codebook(cfg)
        self.codebook.to(device)
        self.k = cfg["model"]["codebook"]["k"]
        self.decoder = Decoder(cfg)
        self.act = torch.nn.Identity()

        self.penalty_weight = 0.25
        if self.cfg["model"]["enable_perceptual"]:
            self.perp_loss_fn = VGGPerceptualLoss()

    def reconstruction_loss(self, x, x_p) -> torch.Tensor:
        return torch.nn.functional.mse_loss(x, x_p)

    @torch.no_grad()
    def generate(self, x):
        x = self.encoder(x)
        x_latent = x.clone()
        x, vq_config = self.codebook(x)
        vq_config["encoder_output"] = x_latent

        x = self.decoder(x)
        x = self.act(x)

        return self._reverse_scale(x), vq_config

    @torch.no_grad()
    def quantize_decoder(self, q_z, x=None):

        x = self.codebook.decode(q_z, x)
        x = self.decoder(x)
        x = self.act(x)
        return self._reverse_scale(x)

    def _reverse_scale(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 + 0.5

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
        # z_h, discrete_h, codebook_loss = self.codebook(z_h)
        z_h, vq_config = self.codebook(z_h)

        x_h = self.act(self.decoder(z_h))
        x_h = self.decoder(z_h)
        reco_loss = self.reconstruction_loss(x, x_h)

        if self.cfg["model"]["enable_perceptual"]:
            perp_loss = self.perp_loss_fn(
                self._reverse_scale(x_h),
                self._reverse_scale(x),
                feature_layers=[0, 1, 2, 3],
                style_layers=[],
            )
        reco_loss += perp_loss
        # reco_loss += usage_loss * self.penalty_weight
        return x_h, (vq_config["loss"], reco_loss), vq_config
