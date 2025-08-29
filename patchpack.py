
import torch
import torch.nn as nn

class PatchPack(nn.Module):
    """2x2 spatial-to-channel reshape using PixelUnshuffle.
    Input:  (B, 3, 608, 608)
    Output: (B, 12, 304, 304)
    """
    def __init__(self, downscale: int = 2):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unshuffle(x)
