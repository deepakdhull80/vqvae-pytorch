import math
import torch

def get_activation(name: str) -> torch.nn.Module:
    if name == 'relu':
        return torch.nn.ReLU()
    elif name == 'tanh':
        return torch.nn.Tanh()
    else:
        raise ValueError("Unknown activation function {}".format(name))


def calculate_conv_output(inp, kernel, stride=1, padding=0):
    return math.floor((inp - kernel + 2*padding)/stride + 1)

def calculate_conv_transpose_output(inp, kernel, stride=1, padding=0):
    return int((inp + kernel + 2*padding) * stride + 1)