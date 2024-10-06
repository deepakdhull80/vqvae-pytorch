import torch

class Codebook(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.k = cfg['model']['codebook']['k']
        self.dim = cfg['model']['codebook']['dim']
        self.lookup_table = torch.nn.Embedding(num_embeddings=self.k, embedding_dim=self.dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w = x.shape
        x = x.view(b, d, -1).permute(0, 2, 1).contiguous()
        # find similar embedding in lookup table
        # do dot product and take topk 1