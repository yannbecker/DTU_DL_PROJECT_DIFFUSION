import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


# Load all 4 datasets
# adata_sc_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/sc_processed_genes.h5ad')
adata_sc_transcripts = ad.read_h5ad('/work3/s193518/scIsoPred/data/sc_processed_transcripts.h5ad')
# adata_bulk_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad')
# adata_bulk_transcripts = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad')

# Store in dictionary for easy iteration
datasets = {
    # 'sc_genes': adata_sc_genes,
    'sc_transcripts': adata_sc_transcripts,
    # 'bulk_genes': adata_bulk_genes,
    # 'bulk_transcripts': adata_bulk_transcripts
}

# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

def explore_anndata(name, adata):
    """Comprehensive exploration of AnnData object"""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}\n")
    
    # Basic shape
    print(f"Shape: {adata.shape[0]} cells/genes × {adata.shape[1]} features")
    print(f"  - n_obs (cells): {adata.n_obs}")
    print(f"  - n_vars (genes/transcripts): {adata.n_vars}")
    
    # Data matrix type
    print(f"\nData Matrix (X):")
    print(f"  - Type: {type(adata.X)}")
    print(f"  - Dtype: {adata.X.dtype}")
    print(f"  - Memory: {adata.X.data.nbytes / 1e9:.2f} GB" if hasattr(adata.X, 'data') else f"  - Memory: {adata.X.nbytes / 1e9:.2f} GB")
    print(f"  - Sparse: {type(adata.X).__name__ == 'csr_matrix' or type(adata.X).__name__ == 'csc_matrix'}")
    
    # Available annotations
    print(f"\nCell metadata (obs): {list(adata.obs.columns)}")
    print(f"Gene/Transcript metadata (var): {list(adata.var.columns)}")
    print(f"Layers: {list(adata.layers.keys()) if adata.layers else 'None'}")
    print(f"Obs metadata (obsm): {list(adata.obsm.keys()) if adata.obsm else 'None'}")
    print(f"Var metadata (varm): {list(adata.varm.keys()) if adata.varm else 'None'}")
    print(f"Unstructured annotations (uns): {list(adata.uns.keys()) if adata.uns else 'None'}")
    
    # Statistical summary
    print(f"\nExpression Statistics:")
    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X
    
    print(f"  - Min: {X_dense.min():.2f}")
    print(f"  - Max: {X_dense.max():.2f}")
    print(f"  - Mean: {X_dense.mean():.2f}")
    print(f"  - Median: {np.median(X_dense):.2f}")
    print(f"  - Sparsity (% zeros): {(X_dense == 0).sum() / X_dense.size * 100:.2f}%")
    
    # Per-cell statistics
    cell_counts = X_dense.sum(axis=1)
    print(f"\nPer-Cell Counts (Library Size):")
    print(f"  - Min: {cell_counts.min():.0f}")
    print(f"  - Max: {cell_counts.max():.0f}")
    print(f"  - Mean: {cell_counts.mean():.0f}")
    print(f"  - Median: {np.median(cell_counts):.0f}")
    
    # Per-gene/transcript statistics
    feature_counts = X_dense.sum(axis=0)
    print(f"\nPer-Feature Total Counts:")
    print(f"  - Min: {feature_counts.min():.0f}")
    print(f"  - Max: {feature_counts.max():.0f}")
    print(f"  - Mean: {feature_counts.mean():.0f}")
    print(f"  - Features with 0 counts: {(feature_counts == 0).sum()}")
    
    # Cell type distribution (if available)
    if 'cell_type' in adata.obs.columns:
        print(f"\nCell Type Distribution:")
        cell_type_counts = adata.obs['cell_type'].value_counts()
        for ct, count in cell_type_counts.items():
            print(f"  - {ct}: {count} ({count/len(adata.obs)*100:.1f}%)")
    
    # Check for batch effects
    if 'batch' in adata.obs.columns:
        print(f"\nBatch Distribution:")
        batch_counts = adata.obs['batch'].value_counts()
        for batch, count in batch_counts.items():
            print(f"  - {batch}: {count}")
    
    return adata

# Run exploration for all datasets
for name, adata in datasets.items():
    explore_anndata(name, adata)




