import math
from functools import partial
from typing import Literal

import ot
import torch
from torch import nn
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import *
import scanpy as sc
import sys, os

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
    
def compute_mmd(original : np.ndarray, generated : np.ndarray, kernel=None) -> float:
    x = torch.from_numpy(generated)
    y = torch.from_numpy(original)
    Loss = MMDLoss()
    return Loss.forward(x,y).numpy()


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
    
def compute_kl(original, generated):
    
    x = torch.from_numpy(generated)
    y = torch.from_numpy(original)
    GaussianKLD = GaussianKLDivergence()
    return GaussianKLD.forward(x,y).numpy()


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

def compute_wasserstein(
        original : np.ndarray,
        generated : np.ndarray, 
        method:Literal["emd", "sinkhorn"] ="emd", 
        reg:float = 0.05, 
        power:int =2 ):

    x = torch.from_numpy(generated)
    y = torch.from_numpy(original)
    return wasserstein(x,y,method,reg,power)


def compute_correlations():
    pass

def compute_random_forest(
        original: np.ndarray, 
        generated: np.ndarray, 
        output_path: str,
        figure_name: str,
        n_estimators: int = 1000, 
        max_depth: int = 5,       
        oob_score: bool = True,
        class_weight: str = "balanced",
        random_state: int = 1):
    # Combine original and generated data
    data = np.concatenate((original, generated), axis=0)
    label = np.concatenate((np.ones(original.shape[0]), np.zeros(generated.shape[0])))
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        data, label,
        test_size=0.25,
        random_state=random_state,
        stratify=label
    )
    # Initialize and train Random Forest Classifier
    rfc1 = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        oob_score=oob_score,
        class_weight=class_weight,
        random_state=random_state
    )
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Fit the model
    rfc1.fit(X_train, y_train)

    rfc1_lab = rfc1.predict(X_train)
    rfc1_pre = rfc1.predict(X_val)
    
    if oob_score:
        print("OOB score of random forest:", rfc1.oob_score_)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, rfc1_lab)
    val_acc = accuracy_score(y_val, rfc1_pre)
    print("Accuracy in training set:", train_acc)
    print("Accuracy in validation set:", val_acc)
    
    # Predict probabilities for validation set
    pre_y = rfc1.predict_proba(X_val)[:, 1]
    
    # Compute ROC curve and AUC
    fpr_Nb, tpr_Nb, _ = roc_curve(y_val, pre_y)
    aucval = auc(fpr_Nb, tpr_Nb)
    print(f"AUC value: {aucval:.4f}")
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
    plt.plot(fpr_Nb, tpr_Nb, "r", linewidth=3, label=f'Random Forest (AUC = {aucval:.4f})')
    
    plt.grid(True)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC Curve of Random Forest: Distinguishing Original vs. Generated Data")
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(output_path, figure_name))
    plt.close()
    
    return {
        "train accuracy": train_acc, 
        "validation accuracy": val_acc, 
        "auc": aucval, 
        "oob_score": rfc1.oob_score_ if oob_score else None, 
        "False Positive Rate": fpr_Nb.tolist(),
        "True Positive Rate": tpr_Nb.tolist()
    }




# ===========================
#   Convenience objects
# ===========================
mmd = MMDLoss()

kl = GaussianKLDivergence()
wasserstein1 = partial(wasserstein, method="sinkhorn", power=1)
wasserstein2 = partial(wasserstein, method="sinkhorn", power=2)

MMD_METRICS = {"mmd": mmd}
KL_METRICS = {"kl": kl}
WASSERSTEIN_METRICS = {"wass1": wasserstein1, "wass2": wasserstein2}
