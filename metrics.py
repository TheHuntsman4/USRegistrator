import torch
import torch.nn.functional as F

def mse(warped, fixed):
    return F.mse_loss(warped, fixed).item()

def mae(warped, fixed):
    return F.l1_loss(warped, fixed).item()

def global_ncc(warped, fixed, eps=1e-5):
    w = warped - warped.mean()
    f = fixed - fixed.mean()
    num = (w * f).sum()
    den = torch.sqrt((w*w).sum() * (f*f).sum() + eps)
    return (num / den).item()

def gradient_l2(ddf):
    # ddf: (B,3,D,H,W)
    dz = torch.diff(ddf, dim=2).pow(2)
    dy = torch.diff(ddf, dim=3).pow(2)
    dx = torch.diff(ddf, dim=4).pow(2)
    return (dz.mean() + dy.mean() + dx.mean()).item()

METRICS = {
    "mse": mse,
    "mae": mae,
    "ncc": global_ncc,
    "grad_l2": gradient_l2,
}
