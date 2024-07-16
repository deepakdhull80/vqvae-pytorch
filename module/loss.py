import torch
import torch.nn.functional as F

class ReconstructionLoss(torch.nn.Module):
    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    
    def forward(self, predict, actual) -> torch.Tensor:
        return F.mse_loss(predict, actual)