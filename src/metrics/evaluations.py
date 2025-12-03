import math
from functools import partial
from typing import Literal

import ot
import torch
from torch import nn


class RBFKernel(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x**2).sum(dim=1, keepdim=True)  # Bx x 1
        y_norm = (y**2).sum(dim=1, keepdim=True)  # By x 1
        squared_ell_2 = x_norm - 2 * x @ y.T + y_norm.T  # Bx x By

        return torch.exp(-self.scale * squared_ell_2)


class BrayCurtisKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.abs(x - y).sum(dim=2)  # Bx x By
        denominator = torch.abs(x + y).sum(dim=2) + 1e-8

        return 1 - numerator / denominator


class TanimotoKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = (x * y).sum(dim=2)  # Bx x By
        denominator = (x + y - x * y).sum(dim=2) + 1e-8

        return numerator / denominator


class RuzickaKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.min(x, y).sum(dim=2)  # Bx x By
        denominator = torch.max(x, y).sum(dim=2) + 1e-8

        return numerator / denominator


class MMDLoss(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.kernel(x, x)
        k_yy = self.kernel(y, y)
        k_xy = self.kernel(x, y)

        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


class GaussianKLDivergence(nn.Module):
    """
    KL( N(mu1, sigma1) || N(mu2, sigma2) )
    where sigma are diagonal covariances (vectors)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # Compute empirical means
        mu1 = x.mean(dim=0)
        mu2 = y.mean(dim=0)

        # Variances (diagonal covariances)
        var1 = x.var(dim=0) + 1e-8
        var2 = y.var(dim=0) + 1e-8

        # KL divergence between two diagonal Gaussians
        kl = 0.5 * (
            torch.log(var2 / var1).sum()
            + (var1 / var2).sum()
            + ((mu2 - mu1) ** 2 / var2).sum()
            - x.shape[1]
        )

        return kl



def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Literal["emd", "sinkhorn"] = "emd",
    reg: float = 0.05,
    power: int = 2,
) -> float:
    assert power == 1 or power == 2

    if method == "emd" or method is None:
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = ot.unif(x0.shape[0], type_as=x0), ot.unif(x1.shape[0], type_as=x1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M, numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret


REGRESSION_METRICS = {
    "mse": mean_squared_error,
    "pcc": pearson_corrcoef,
    # "scc": spearman_corrcoef,
    # "r2": partial(r2_score, multioutput="raw_values"),
}

MMD_METRICS = {
    "mmd_braycurtis_counts": MMDLoss(kernel=BrayCurtisKernel()),
    "mmd_tanimoto": MMDLoss(kernel=TanimotoKernel()),
    "mmd_ruzicka_counts": MMDLoss(kernel=RuzickaKernel()),
    "mmd_rbf": MMDLoss(kernel=RBFKernel()),
}

WASSERSTEIN_METRICS = {
    "wasserstein1_sinkhorn": partial(wasserstein, method="sinkhorn", power=1),
    "wasserstein2_sinkhorn": partial(wasserstein, method="sinkhorn", power=2),
}


R2_METRICS = {
    "r2_mean": lambda preds, target: r2_score(preds.mean(0), target.mean(0)),
    "r2_var": lambda preds, target: r2_score(preds.var(0), target.var(0)),
}

KL_METRICS = {
    "gaussian_kl": GaussianKLDivergence(),
}

