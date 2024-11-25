import torch
from typing import Union

# from vqtorch.nn import VectorQuant

from module.vqvae.encoder import ImageEncoder
from module.vqvae.codebook import Codebook
from module.vqvae.decoder import Decoder
from module.loss import VGGPerceptualLoss


class VQVAE(torch.nn.Module):
    def __init__(self, cfg: dict, device: Union[str, torch.device] = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg)
        self.codebook = Codebook(cfg)
        self.codebook.to(device)
        self.decoder = Decoder(cfg)

        self.pre_quant_layer = torch.nn.Conv2d(**cfg["model"]["pre_codebook"]["param"])
        if "post_codebook" in cfg["model"]:
            self.post_quant_layer = torch.nn.Conv2d(
                **cfg["model"]["post_codebook"]["param"]
            )
        else:
            self.post_quant_layer = torch.nn.Identity()
        # self.act = torch.nn.Tanh()
        self.act = torch.nn.Identity()

        self.mu = (
            torch.tensor(self.cfg["data"]["transform"]["normalize"]["mean"])
            .view(1, -1, 1, 1)
            .to(device)
        )
        self.std = (
            torch.tensor(self.cfg["data"]["transform"]["normalize"]["std"])
            .view(1, -1, 1, 1)
            .to(device)
        )

        self.penalty_weight = 0.25

        if self.cfg["model"]["enable_perceptual"]:
            self.perp_loss_fn = VGGPerceptualLoss()

    def reconstruction_loss(self, x, x_p) -> torch.Tensor:
        return torch.nn.functional.mse_loss(x, x_p)

    @torch.no_grad()
    def generate(self, x):
        x = self.encoder(x)
        x = self.pre_quant_layer(x)

        x_latent = x.clone()
        x, vq_config = self.codebook(x)
        x = self.codebook.decode(vq_config["q"])
        vq_config["encoder_output"] = x_latent
        x = self.post_quant_layer(x)
        x = self.decoder(x)
        x = self.act(x)

        return self._reverse_scale(x), vq_config

    @torch.no_grad()
    def quantize_decoder(self, q_z, x=None):

        x = self.codebook.decode(q_z, x)
        self.post_quant_layer(x)
        x = self.decoder(x)
        x = self.act(x)
        return self._reverse_scale(x)

    def _reverse_scale(self, x: torch.Tensor) -> torch.Tensor:
        # return x * 0.5 + 0.5

        return x * self.std + self.mu

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
        z_h = self.pre_quant_layer(z_h)
        # z_h, discrete_h, codebook_loss = self.codebook(z_h)
        z_h, vq_config = self.codebook(z_h)
        z_h = self.post_quant_layer(z_h)
        x_h = self.act(self.decoder(z_h))

        reco_loss = torch.nn.functional.mse_loss(x_h, x)

        perp_loss = 0
        if self.cfg["model"]["enable_perceptual"]:
            perp_loss = self.perp_loss_fn(
                self._reverse_scale(x_h),
                self._reverse_scale(x),
                feature_layers=[0, 1, 2, 3],
                style_layers=[],
            )
        reco_loss += perp_loss
        # reco_loss += usage_loss * self.penalty_weight
        return x_h, (vq_config["loss"], reco_loss, vq_config)
