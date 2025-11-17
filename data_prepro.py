import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


# Load all 4 datasets
adata_sc_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/sc_processed_genes.h5ad')
adata_sc_transcripts = ad.read_h5ad('/work3/s193518/scIsoPred/data/sc_processed_transcripts.h5ad')
adata_bulk_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad')
adata_bulk_transcripts = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad')

# Store in dictionary for easy iteration
datasets = {
    'sc_genes': adata_sc_genes,
    'sc_transcripts': adata_sc_transcripts,
    'bulk_genes': adata_bulk_genes,
    'bulk_transcripts': adata_bulk_transcripts
}


# 1. How many multi-isoform genes?
print(f"Multi-isoform genes: {len(ad.uns['multi_isoform_genes'])}")
print(f"Total genes: {len(ad.uns['gene_n_transcripts'])}")

# 2. Is data raw counts?
if hasattr(ad.X, 'toarray'):
    X_sample = ad.X[:100, :100].toarray()
else:
    X_sample = ad.X[:100, :100]
print(f"Is integer data: {np.allclose(X_sample, X_sample.astype(int))}")

# 3. How many cell clusters?
print(f"Number of clusters: {ad.obs['leiden'].nunique()}")

# 4. Data size
print(f"Shape: {ad.shape[0]} cells × {ad.shape[1]} isoforms")

# 5. Sparsity
if hasattr(ad.X, 'toarray'):
    sparsity = (ad.X.toarray() == 0).mean() * 100
else:
    sparsity = (ad.X == 0).mean() * 100
print(f"Sparsity: {sparsity:.1f}%")


