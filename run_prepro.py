import os 
import numpy as np
import scanpy as sc
import anndata as ad
from src.normalize import normalize_bulk_data
from scimilarity import CellAnnotation
from scimilarity.utils import lognorm_counts, align_dataset


adata_bulk_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad')

adata_bulk_genes.layers['counts'] = adata_bulk_genes.X.copy()


adata_bulk_transcripts = normalize_bulk_data(adata_bulk_genes)

# Align dataset to scimilarity's expected gene set
adata_bulk_genes = align_dataset(adata_bulk_genes)

# Initialize CellAnnotation with pretrained model
# Download model from https://zenodo.org/records/10685499
model_path = 'path/to/model_v1.0.h5'

ca = CellAnnotation(
    model_path=model_path,
    device='cuda'  # or 'cpu'
)

# Get embeddings (128-dimensional latent space)
embeddings = ca.get_embeddings(adata_bulk_genes)

# Add embeddings to adata
adata_bulk_genes.obsm['X_scimilarity'] = embeddings

# Save
adata_bulk_genes.write_h5ad('bulk_genes_with_embeddings.h5ad')

print(f"Generated embeddings shape: {embeddings.shape}")