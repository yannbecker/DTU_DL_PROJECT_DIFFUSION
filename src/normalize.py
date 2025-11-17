import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

def normalize_bulk_data(adata, target_sum=1e4):
    """
    Normalize bulk RNA-seq data with library size normalization and log1p transform
    """
    # Library size normalization
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Log1p transformation
    sc.pp.log1p(adata)
    
    return adata
