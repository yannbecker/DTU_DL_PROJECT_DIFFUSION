import math
from functools import partial
from typing import Literal

import ot
import torch
from torch import nn


# ==========
#  MMD
# ==========
class RBFKernel(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x**2).sum(dim=1, keepdim=True)
        y_norm = (y**2).sum(dim=1, keepdim=True)
        sq_dist = x_norm - 2 * x @ y.T + y_norm.T
        return torch.exp(-self.scale * sq_dist)


class MMDLoss(nn.Module):
    def __init__(self, kernel=None):
        super().__init__()
        self.kernel = kernel or RBFKernel()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.kernel(x, x).mean()
        k_yy = self.kernel(y, y).mean()
        k_xy = self.kernel(x, y).mean()
        return k_xx + k_yy - 2 * k_xy


# ===========================
#   KL divergence (Gaussian)
# ===========================
class GaussianKLDivergence(nn.Module):
    """
    KL divergence between two diagonal Gaussians fitted to samples x and y.
    """
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu1, mu2 = x.mean(0), y.mean(0)
        var1 = x.var(0) + 1e-8
        var2 = y.var(0) + 1e-8

        kl = 0.5 * (
            torch.log(var2 / var1).sum()
            + (var1 / var2).sum()
            + ((mu2 - mu1)**2 / var2).sum()
            - x.shape[1]
        )
        return kl


# ===========================
#   Wasserstein distance
# ===========================
def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Literal["emd", "sinkhorn"] = "emd",
    reg: float = 0.05,
    power: int = 2,
):
    assert power in (1, 2)

    ot_fn = (
        ot.emd2 if method == "emd"
        else partial(ot.sinkhorn2, reg=reg)
    )

    a = ot.unif(x0.shape[0], type_as=x0)
    b = ot.unif(x1.shape[0], type_as=x1)

    M = torch.cdist(x0, x1)
    if power == 2:
        M = M ** 2

    dist = ot_fn(a, b, M, numItermax=int(1e7))

    if power == 2:
        dist = math.sqrt(dist)

    return dist


# ===========================
#   Convenience objects
# ===========================
mmd = MMDLoss()
kl = GaussianKLDivergence()
wasserstein1 = partial(wasserstein, method="sinkhorn", power=1)
wasserstein2 = partial(wasserstein, method="sinkhorn", power=2)
