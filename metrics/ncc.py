import torch
from torch import Tensor
try:
    from monai.metrics import RegressionMetric
    from monai.losses import LocalNormalizedCrossCorrelationLoss
except ImportError:
    from monai.metrics import Metric
    RegressionMetric = Metric

def global_ncc(warped: Tensor, fixed: Tensor, eps: float = 1e-5) -> float:
    """
    Compute Global Normalized Cross Correlation.
    """
    w = warped - warped.mean()
    f = fixed - fixed.mean()
    num = (w * f).sum()
    den = torch.sqrt((w*w).sum() * (f*f).sum() + eps)
    return (num / den).item()

class NCC(RegressionMetric):
    """
    Normalized Cross Correlation Metric.
    Wraps the functional implementation or MONAI's loss if appropriate.
    Here we implement the global NCC as requested.
    """
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, y_pred: Tensor, y: Tensor) -> Tensor:
        # compute for batch
        # This returns a tensor, not a float item, to be consistent with usage as a metric class
        return torch.tensor(global_ncc(y_pred, y, self.eps))

    def aggregate(self):
        # Implement accumulation logic if needed, or rely on base class
        pass
