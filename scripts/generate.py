import os
import sys
import mmh3
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Lambda, Resize, ToPILImage
from typing import Dict, List

root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder)

from module.vae import AutoEncoder, VariationalAutoEncoder
from module.constant import ModelEnum


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ModelHelper(metaclass=Singleton):

    def __init__(self, cfg: Dict, model_chkpt_path: str, device: str) -> None:
        self.cfg = cfg
        if ModelEnum.AUTO_ENCODER.value == cfg["model"]["name"]:
            self.model = AutoEncoder(cfg)
        elif ModelEnum.VARIATIONAL_AUTO_ENCODER.value == cfg["model"]["name"]:
            self.model = VariationalAutoEncoder(cfg)
        else:
            raise ValueError(
                f"Model {cfg['model']['name']} not supported. Available models are {ModelEnum.list()}"
            )
        self.device = device
        self.model_chkpt_path = model_chkpt_path
        _state_dict = torch.load(model_chkpt_path, map_location="cpu")
        print(self.model.load_state_dict(_state_dict))
        self.model.eval()
        self.model = self.model.to(self.device)

        self.transform = Compose(
            [
                Lambda(lambda batch: torch.clamp_(batch * 0.5 + 0.5, min=0, max=1)),
                Resize((self.cfg["img_shape"], self.cfg["img_shape"]), antialias=True),
            ]
        )

        self.to_pil = ToPILImage()

    def refresh_weights(self, model: torch.nn.Module):
        self.model.load_state_dict(model.state_dict())
        self.model.to(self.device)

    @torch.no_grad()
    def generate_image(self, n_items: int = 4) -> List[Image.Image]:

        latent_dim = self.cfg["model"]["encoder"]["fc"][-1] * (
            1 if ModelEnum.AUTO_ENCODER.value == self.cfg["model"]["name"] else 2
        )
        z = torch.randn((n_items, latent_dim))

        img = self.model.decoder(z.to(self.device))
        return [self.to_pil(img) for img in self.transform(img).unbind(0)]
    
    @torch.no_grad()
    def generate_conditional_image(self, n_items: int = 4) -> List[Image.Image]:

        latent_dim = self.cfg["model"]["encoder"]["fc"][-1]
        z = torch.randn((n_items, latent_dim))
        z = torch.concat([z, torch.nn.functional.one_hot(torch.randint(low=0, high=self.cfg['data']['num_classes']-1, size=(n_items,)), num_classes=self.cfg['data']['num_classes'])], dim=1)
        img = self.model.decoder(z.to(self.device))
        return [self.to_pil(img) for img in self.transform(img).unbind(0)]

    def generate_and_save(
        self, n_items: int = 4, output: str = "output", file_prefix: int = 0
    ) -> None:
        os.makedirs(output, exist_ok=True)

        imgs = self.generate_conditional_image(n_items=n_items) if self.cfg['data']['clz'] == 'COCOConditionalDataset' else self.generate_image(n_items=n_items)
        
        for img in imgs:
            file_name = str(
                mmh3.hash_from_buffer(np.random.rand(int(1e5)), signed=False)
            )
            img.save(f"{output}/{file_prefix}-{file_name}.jpg")

    @torch.no_grad()
    def get_latent_emb(self, img: torch.Tensor) -> torch.Tensor:
        z = self.model.encoder(img.to(self.device))
        return z.cpu()


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", required=True)
    parser.add_argument("-d", "--device", dest="device", default="cpu")
    parser.add_argument("-p", "--chkpt-path", dest="chkpt_path", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"config/{args.config}.yaml", "r"))

    model_helper = ModelHelper(
        cfg, model_chkpt_path=args.chkpt_path, device=args.device
    )
    model_helper.generate_and_save(n_items=2)
