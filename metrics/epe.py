import torch
from torch import Tensor
from monai.metrics import Metric

def epe(ddf: Tensor, gt_dvf: Tensor, eps: float = 1e-9) -> float:
    """
    Endpoint Error (EPE) between predicted ddf and ground-truth dvf.

    ddf, gt_dvf: (B, 3, D, H, W)
    Returns: Euclidean distance averaged over all batch and spatial dimensions.
    """
    if ddf.shape != gt_dvf.shape:
        raise ValueError(f"EPE: shape mismatch {ddf.shape} vs {gt_dvf.shape}")
    diff = ddf - gt_dvf              # (B,3,D,H,W)
    sq = diff.pow(2).sum(dim=1)      # (B,D,H,W)
    dist = torch.sqrt(sq + eps)      # (B,D,H,W)
    return dist.mean().item()

class EPE(Metric):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def __call__(self, ddf: Tensor, gt_dvf: Tensor) -> Tensor:
        return torch.tensor(epe(ddf, gt_dvf, self.eps))
