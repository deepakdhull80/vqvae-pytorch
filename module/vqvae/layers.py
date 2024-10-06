import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.conv_blk_1 = torch.nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv_blk_2 = torch.nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        return self.conv_blk_2(self.relu(self.conv_blk_1(x))) + x