from copy import deepcopy
import torch

from module.vqvae.layers import ResidualBlock


class ImageEncoder(torch.nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        layer_cfg_li = deepcopy(cfg["model"]["encoder"]["layers"])
        layers = []

        for layer_cfg in layer_cfg_li:
            if layer_cfg["name"] == "conv2d":
                layers.append(torch.nn.Conv2d(**layer_cfg["param"]))
            elif layer_cfg["name"] == "resnet":
                for _ in range(layer_cfg["repeat"]):
                    layers.append(ResidualBlock(dim=layer_cfg["dim"]))
            else:
                raise NotImplementedError(f"{layer_cfg['name']} not implemented yet!")

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
