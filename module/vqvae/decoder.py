from copy import deepcopy
import torch

from module.vqvae.layers import ResidualStack


class Decoder(torch.nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        layer_cfg_li = deepcopy(cfg["model"]["decoder"]["layers"])
        layers = []

        for i, layer_cfg in enumerate(layer_cfg_li):
            if layer_cfg["name"] == "conv2d_transpose":
                layers.append(torch.nn.ConvTranspose2d(**layer_cfg["param"]))
            elif layer_cfg["name"] == "conv2d":
                layers.append(torch.nn.Conv2d(**layer_cfg["param"]))
            elif layer_cfg["name"] == "resnet":
                self._residual_stack = ResidualStack(**layer_cfg["param"])
            else:
                raise NotImplementedError(f"{layer_cfg['name']} not implemented yet!")

            if "activation" in layer_cfg:
                if layer_cfg["activation"] == "ReLU":
                    layers.append(torch.nn.ReLU())
                else:
                    raise NotImplementedError(f"{layer_cfg['name']}")

            if "norm" in layer_cfg:
                if layer_cfg["norm"] == "BN":
                    layers.append(
                        torch.nn.BatchNorm2d(layer_cfg["param"]["out_channels"])
                    )
                else:
                    raise NotImplementedError(f"{layer_cfg['norm']}")

            if "dropout" in layer_cfg:
                layers.append(torch.nn.Dropout(layer_cfg["dropout"]))

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
