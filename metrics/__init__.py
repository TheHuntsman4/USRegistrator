from .regression import MSEMetric, MAEMetric, mse, mae
from .ncc import NCC, global_ncc
from .smoothness import GradientL2, gradient_l2
from .epe import EPE, epe

METRICS = {
    "mse": mse,
    "mae": mae,
    "ncc": global_ncc,
    "grad_l2": gradient_l2,
    "epe": epe,
}

# Also expose class-based mapping if needed, or let user access classes directly
