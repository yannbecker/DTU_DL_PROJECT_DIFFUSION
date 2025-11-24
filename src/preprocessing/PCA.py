# Imports
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_pca(adata, n_components=1024, plot=True):
    """
    Run PCA on adata.X and return reduced numpy array (cells x n_components).
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    # Optionally center / standardize here if needed

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if plot:
        explained = pca.explained_variance_ratio_
        cumsum = np.cumsum(explained)
        components_range = np.arange(1, len(explained) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(components_range, cumsum, marker="o", linewidth=2)
        plt.axhline(0.90, color="grey", linestyle="--",
                    label="90% threshold")
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative explained variance ratio")
        plt.title("Cumulative explained variance")
        plt.grid(True)
        plt.legend()
        plt.show()

    return X_pca, pca




