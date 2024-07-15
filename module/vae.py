import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.functions import get_activation, calculate_conv_output, calculate_conv_transpose_output

class Encoder(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        layers = []
        inp_channel = 3
        final_shape = cfg['img_shape']
        
        for i in range(len(cfg['model']['encoder']['conv_channels'])):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=inp_channel, 
                        out_channels=cfg['model']['encoder']['conv_channels'][i], 
                        kernel_size=cfg['model']['encoder']['kernels'][i], 
                        stride=cfg['model']['encoder']['strides'][i],
                        bias=False
                        ),
                    get_activation(cfg['model']['encoder']['activation'][i]),
                    nn.BatchNorm2d(cfg['model']['encoder']['conv_channels'][i]) if cfg['model']['encoder']['norm'][i] else nn.Identity()                    
                )
            )
            inp_channel = cfg['model']['encoder']['conv_channels'][i]
            final_shape = calculate_conv_output(
                final_shape, 
                kernel=cfg['model']['encoder']['kernels'][i],
                stride=cfg['model']['encoder']['strides'][i]
            )
        
        self.layers = nn.Sequential(*layers)
        self.flatten_size = fc_inp = final_shape**2 * inp_channel
        
        fc_layers = []
        for fc_output in cfg['model']['encoder']['fc']:
            fc_layers.append(
                nn.Sequential(
                    nn.Linear(fc_inp, fc_output, bias=True),
                    nn.LayerNorm(fc_output),
                    nn.ReLU()
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
        inp = cfg['model']['latent_dim']
        for i in range(len(cfg['model']['encoder']['fc'])-1, -1, -1):
            out = encoder_flatten_size if i == 0 else cfg['model']['encoder']['fc'][i-1]
            fc_layers.append(
                nn.Sequential(
                    nn.Linear(cfg['model']['encoder']['fc'][i], out, bias=True),
                    nn.LayerNorm(out),
                    nn.ReLU()
                )
            )
        self.fc = nn.Sequential(*fc_layers)
        
        layers = []
        self.final_conv_channels = inp_channel = cfg['model']['encoder']['conv_channels'][-1]
        self.latent_space_img_size = int(math.sqrt(encoder_flatten_size / self.final_conv_channels))
        final_shape = self.latent_space_img_size
        
        for i in range(len(cfg['model']['decoder']['conv_channels'])):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=inp_channel, 
                        out_channels=cfg['model']['decoder']['conv_channels'][i], 
                        kernel_size=cfg['model']['decoder']['kernels'][i], 
                        stride=cfg['model']['decoder']['strides'][i],
                        bias=False
                        ),
                    get_activation(cfg['model']['decoder']['activation'][i]),
                    nn.BatchNorm2d(cfg['model']['decoder']['conv_channels'][i]) if cfg['model']['decoder']['norm'][i] else nn.Identity()                    
                )
            )
            inp_channel = cfg['model']['decoder']['conv_channels'][i]
            final_shape = calculate_conv_transpose_output(
                final_shape, 
                kernel=cfg['model']['decoder']['kernels'][i],
                stride=cfg['model']['decoder']['strides'][i]
            )
        
        self.layers = nn.Sequential(*layers)
        # self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        b, _ = x.shape
        x = self.fc(x)
        x = x.view(b, self.final_conv_channels, self.latent_space_img_size, self.latent_space_img_size).contiguous()
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x = self.decoder(z)
        return x

class VariationalAutoEncoder(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        
    def forward(self, x):
        return x



