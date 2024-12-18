import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.functions import (
    get_activation,
    calculate_conv_output,
    calculate_conv_transpose_output,
)


class Encoder(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        layers = []
        inp_channel = 3
        final_shape = cfg["img_shape"]

        for i in range(len(cfg["model"]["encoder"]["conv_channels"])):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=inp_channel,
                        out_channels=cfg["model"]["encoder"]["conv_channels"][i],
                        kernel_size=cfg["model"]["encoder"]["kernels"][i],
                        stride=cfg["model"]["encoder"]["strides"][i],
                        bias=False,
                    ),
                    get_activation(cfg["model"]["encoder"]["activation"][i]),
                    (
                        nn.BatchNorm2d(cfg["model"]["encoder"]["conv_channels"][i])
                        if cfg["model"]["encoder"]["norm"][i]
                        else nn.Identity()
                    ),
                )
            )
            inp_channel = cfg["model"]["encoder"]["conv_channels"][i]
            final_shape = calculate_conv_output(
                final_shape,
                kernel=cfg["model"]["encoder"]["kernels"][i],
                stride=cfg["model"]["encoder"]["strides"][i],
            )

        self.layers = nn.Sequential(*layers)
        self.flatten_size = fc_inp = final_shape**2 * inp_channel

        fc_layers = []
        for fc_output in cfg["model"]["encoder"]["fc"]:
            fc_layers.append(
                nn.Sequential(
                    nn.Linear(fc_inp, fc_output, bias=True),
                    nn.LayerNorm(fc_output),
                    nn.ReLU(),
                )
            )
            fc_inp = fc_output

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, cfg: Dict, encoder_flatten_size: int):
        super().__init__()
        self.cfg = cfg
        self.encoder_flatten_size = encoder_flatten_size

        fc_layers = []
        inp = cfg["model"]["latent_dim"]
        for i in range(len(cfg["model"]["encoder"]["fc"]) - 1, -1, -1):
            out = (
                encoder_flatten_size if i == 0 else cfg["model"]["encoder"]["fc"][i - 1]
            )
            _in = cfg["model"]["encoder"]["fc"][i]
            fc_layers.append(
                nn.Sequential(
                    nn.Linear(_in, out, bias=True),
                    nn.LayerNorm(out),
                    nn.ReLU(),
                )
            )
        self.fc = nn.Sequential(*fc_layers)

        layers = []
        self.final_conv_channels = inp_channel = cfg["model"]["encoder"][
            "conv_channels"
        ][-1]
        self.latent_space_img_size = int(
            math.sqrt(encoder_flatten_size / self.final_conv_channels)
        )
        final_shape = self.latent_space_img_size

        for i in range(len(cfg["model"]["decoder"]["conv_channels"])):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=inp_channel,
                        out_channels=cfg["model"]["decoder"]["conv_channels"][i],
                        kernel_size=cfg["model"]["decoder"]["kernels"][i],
                        stride=cfg["model"]["decoder"]["strides"][i],
                        bias=False,
                    ),
                    get_activation(cfg["model"]["decoder"]["activation"][i]),
                    (
                        nn.BatchNorm2d(cfg["model"]["decoder"]["conv_channels"][i])
                        if cfg["model"]["decoder"]["norm"][i]
                        else nn.Identity()
                    ),
                )
            )
            inp_channel = cfg["model"]["decoder"]["conv_channels"][i]
            final_shape = calculate_conv_transpose_output(
                final_shape,
                kernel=cfg["model"]["decoder"]["kernels"][i],
                stride=cfg["model"]["decoder"]["strides"][i],
            )

        self.layers = nn.Sequential(*layers)
        # self.layers = nn.ModuleList(layers)

    def forward(self, x):
        b, _ = x.shape
        x = self.fc(x)
        x = x.view(
            b,
            self.final_conv_channels,
            self.latent_space_img_size,
            self.latent_space_img_size,
        ).contiguous()
        # debug code
        # for mod in self.layers:
        #     x = mod(x)
        #     print("layer output:", x.shape)

        x = self.layers(x)
        return x


class VariationDecoder(nn.Module):
    def __init__(self, cfg: Dict, encoder_flatten_size: int):
        super().__init__()
        self.cfg = cfg
        self.encoder_flatten_size = encoder_flatten_size

        fc_layers = []
        inp = cfg["model"]["latent_dim"]
        for i in range(len(cfg["model"]["encoder"]["fc"]) - 1, -1, -1):
            out = (
                encoder_flatten_size if i == 0 else cfg["model"]["encoder"]["fc"][i - 1]
            )
            _in = (
                cfg["model"]["encoder"]["fc"][i] + cfg["data"]["num_classes"]
                if len(cfg["model"]["encoder"]["fc"]) == i + 1
                else 1
            )
            fc_layers.append(
                nn.Sequential(
                    nn.Linear(_in, out, bias=True),
                    nn.LayerNorm(out),
                    nn.ReLU(),
                )
            )
        self.fc = nn.Sequential(*fc_layers)

        layers = []
        self.final_conv_channels = inp_channel = cfg["model"]["encoder"][
            "conv_channels"
        ][-1]
        self.latent_space_img_size = int(
            math.sqrt(encoder_flatten_size / self.final_conv_channels)
        )
        final_shape = self.latent_space_img_size

        for i in range(len(cfg["model"]["decoder"]["conv_channels"])):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=inp_channel,
                        out_channels=cfg["model"]["decoder"]["conv_channels"][i],
                        kernel_size=cfg["model"]["decoder"]["kernels"][i],
                        stride=cfg["model"]["decoder"]["strides"][i],
                        bias=False,
                    ),
                    get_activation(cfg["model"]["decoder"]["activation"][i]),
                    (
                        nn.BatchNorm2d(cfg["model"]["decoder"]["conv_channels"][i])
                        if cfg["model"]["decoder"]["norm"][i]
                        else nn.Identity()
                    ),
                )
            )
            inp_channel = cfg["model"]["decoder"]["conv_channels"][i]
            final_shape = calculate_conv_transpose_output(
                final_shape,
                kernel=cfg["model"]["decoder"]["kernels"][i],
                stride=cfg["model"]["decoder"]["strides"][i],
            )

        self.layers = nn.Sequential(*layers)
        # self.layers = nn.ModuleList(layers)

    def forward(self, x):
        b, _ = x.shape
        x = self.fc(x)
        x = x.view(
            b,
            self.final_conv_channels,
            self.latent_space_img_size,
            self.latent_space_img_size,
        ).contiguous()
        # debug code
        # for mod in self.layers:
        #     x = mod(x)
        #     print("layer output:", x.shape)

        x = self.layers(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg, self.encoder.flatten_size)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x = self.decoder(z)
        return x, torch.randn(1)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.mu_fc = nn.Linear(
            self.cfg["model"]["latent_dim"], self.cfg["model"]["latent_dim"], bias=False
        )
        self.log_var_fc = nn.Linear(
            self.cfg["model"]["latent_dim"], self.cfg["model"]["latent_dim"], bias=False
        )
        self.decoder = VariationDecoder(cfg, self.encoder.flatten_size)

    def reparameterization(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        mu = self.mu_fc(z)
        log_var = self.log_var_fc(z)
        # epsilon = torch.rand_like(z).to(device)
        sigma = torch.exp(0.5 * log_var)
        # z = mu + sigma * epsilon
        q = torch.distributions.Normal(loc=mu, scale=sigma)
        z = q.rsample()

        # kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        kl_loss = self.kl_divergence(z, mu, sigma)
        return z, kl_loss

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        z = self.encoder(x)
        z, kl_loss = self.reparameterization(z)
        if y is None:
            y = torch.rand_like(z).to(device)
        z = torch.concat([z, y], dim=1)
        x = self.decoder(z)
        return x, kl_loss
