# Imports
import os
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def run_pca(adata, n_components=None, threshold=0.90, plot=True, plot_path='output/plots/pca_variance.png'):
    """
    Run PCA on adata.X with automatic selection of components for threshold variance.

    Args:
        adata: AnnData object with data in .X
        n_components: int or None
            Number of PCA components. If None, selects minimum components
            such that cumulative explained variance >= threshold.
        threshold: float
            Cumulative explained variance threshold to choose number of components if n_components is None.
        plot: bool
            Whether to save the cumulative explained variance plot.
        plot_path: str
            File path to save the plot.

    Returns:
        X_pca: np.ndarray of shape (n_cells, n_components chosen)
        pca: trained sklearn PCA instance
    """
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # Fit PCA with enough components to cover 100% variance initially
    max_components = min(X.shape[0], X.shape[1])
    pca_full = PCA(n_components=max_components)
    pca_full.fit(X)

    explained_cumsum = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is None:
        # Number of components to reach variance threshold
        n_components = int(np.searchsorted(explained_cumsum, threshold)) + 1
        print(f"Automatically selected {n_components} components to reach {int(threshold*100)}% explained variance")
    else:
        print(f"Using user-defined n_components = {n_components}")

    # Refit PCA with selected number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if plot:
    # Ensure directory exists
        output_dir = os.path.dirname(plot_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    comps = np.arange(1, len(explained_cumsum) + 1)
    plt.plot(comps, explained_cumsum, marker='o', linewidth=2, label="Cumulative Explained Variance")
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{int(threshold*100)}% Threshold')
    plt.axvline(x=n_components, color='g', linestyle='--', label=f'Selected Components: {n_components}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved PCA explained variance plot to {plot_path}")

    return X_pca, pca




