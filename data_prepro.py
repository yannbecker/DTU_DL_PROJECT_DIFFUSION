import numpy as np
import pandas as pd
import anndata as ad

adata_sc_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/sc_processed_genes.h5ad')
# adata_sc_transcripts = ad.read_h5ad('/work3/s193518/scIsoPred/data/sc_processed_transcripts.h5ad')
# adata_bulk_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad')
# adata_bulk_transcripts = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad')


print(adata_sc_genes)