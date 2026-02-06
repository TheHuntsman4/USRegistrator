from typing import Union, Optional
import torch
from torch import Tensor
import warnings

try:
    from monai.metrics import MSEMetric, MAEMetric
    has_monai_regression = True
except ImportError:
    from monai.metrics import Metric
    has_monai_regression = False
    
    # Fallback/Standalone implementations if MONAI specific metrics aren't found 
    # or to ensure simple behavior if the user wanted functions. 
    # But user asked to use MONAI if available.
    
    class MSEMetric(Metric):
        def __init__(self, reduction: Union[str, None] = "mean") -> None:
            super().__init__()
            self.reduction = reduction
            self.sq_error_sum = 0.0
            self.count = 0

        def __call__(self, y_pred: Tensor, y: Tensor) -> Tensor:
            # Defines the computation for a batch
            loss = torch.nn.functional.mse_loss(y_pred, y, reduction=self.reduction or 'none')
            return loss

    class MAEMetric(Metric):
        def __init__(self, reduction: Union[str, None] = "mean") -> None:
            super().__init__()
            self.reduction = reduction

        def __call__(self, y_pred: Tensor, y: Tensor) -> Tensor:
             loss = torch.nn.functional.l1_loss(y_pred, y, reduction=self.reduction or 'none')
             return loss

def mse(y_pred: Tensor, y: Tensor) -> float:
    """Functional interface for MSE"""
    return torch.nn.functional.mse_loss(y_pred, y).item()

def mae(y_pred: Tensor, y: Tensor) -> float:
    """Functional interface for MAE"""
    return torch.nn.functional.l1_loss(y_pred, y).item()
