from src.normalize import normalize_bulk_data
import anndata as ad
from scimilarity.utils import load_annotation_model


adata_bulk_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad')

adata_bulk_genes.layers['counts'] = adata_bulk_genes.X.copy()


adata_bulk_transcripts = normalize_bulk_data(adata_bulk_genes)

model_path = 'path/to/annotation_model_v1.pt'
model = load_annotation_model(model_path)

# Extract embeddings using scimilarity encoder
embeddings = model.get_embeddings(adata_bulk_genes)
# embeddings shape: (n_samples, 128)

# Add embeddings to adata
adata_bulk_genes.obsm['X_scimilarity'] = embeddings

# Save for later use
adata_bulk_genes.write_h5ad('bulk_genes_with_embeddings.h5ad')