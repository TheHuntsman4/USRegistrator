import torch
from torch import Tensor
from monai.metrics import Metric

def gradient_l2(ddf: Tensor) -> float:
    """
    Compute Gradient L2 norm for the deformation field.
    ddf: (B,3,D,H,W)
    """
    # ddf: (B,3,D,H,W)
    dz = torch.diff(ddf, dim=2).pow(2)
    dy = torch.diff(ddf, dim=3).pow(2)
    dx = torch.diff(ddf, dim=4).pow(2)
    return (dz.mean() + dy.mean() + dx.mean()).item()

class GradientL2(Metric):
    def __init__(self):
        super().__init__()
    
    def __call__(self, ddf: Tensor) -> Tensor:
        return torch.tensor(gradient_l2(ddf))
