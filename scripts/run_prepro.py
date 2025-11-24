import os 
import numpy as np
import scanpy as sc
import anndata as ad
from scimilarity import CellAnnotation
from scimilarity.utils import lognorm_counts, align_dataset
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing.cell_datasets_loader import load_data

data_path = '/work3/s193518/scIsoPred/data/bulk_processed_transcript.h5ad'

adata_bulk_transcript = load_data(data_dir=data_path)







# # Add embeddings to adata
# adata_bulk_genes.obsm['X_scimilarity'] = embeddings

# # Save
# adata_bulk_genes.write_h5ad('bulk_genes_with_embeddings.h5ad')

# print(f"Generated embeddings shape: {embeddings.shape}")