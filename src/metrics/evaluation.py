from functools import partial
from typing import Literal
from torch import nn
import anndata as ad
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import *
import scanpy as sc
import sys, os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import wasserstein_distance, pearsonr
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA



# Force CUDA device 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ==========================================
# 1. METRICS FUNCTIONS
# ==========================================

############################  MMD  ############################

def compute_mmd(x, y, gamma=1.0):
    x_kernel = rbf_kernel(x, x, gamma=gamma)
    y_kernel = rbf_kernel(y, y, gamma=gamma)
    xy_kernel = rbf_kernel(x, y, gamma=gamma)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

########################  WASSERSTEIN  ########################

def compute_wasserstein(x, y):
    wd_list = [wasserstein_distance(x[:, i], y[:, i]) for i in range(x.shape[1])]
    return float(np.mean(wd_list))

########################  CORRELATIONS  ########################

def compute_correlations(real_data, gen_data):
    mean_real = np.mean(real_data, axis=0)
    mean_gen = np.mean(gen_data, axis=0)
    var_real = np.var(real_data, axis=0)
    var_gen = np.var(gen_data, axis=0)
    # Replace NaNs (e.g. zero-variance genes) to avoid crashes
    mean_real = np.nan_to_num(mean_real)
    mean_gen = np.nan_to_num(mean_gen)
    var_real = np.nan_to_num(var_real)
    var_gen = np.nan_to_num(var_gen)
    corr_mean, _ = pearsonr(mean_real, mean_gen)
    corr_var, _ = pearsonr(var_real, var_gen)
    return float(corr_mean), float(corr_var)

########################  RANDOM FOREST  ########################

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

########################  KL DIVERGENCE  ########################

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