import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats, sparse

def visualize_data_distributions(adata_genes, adata_transcripts, output_dir='./plots'):
    """
    Create comprehensive visualization of data distributions
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Library size distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(adata_genes.obs['total_counts'], bins=50, edgecolor='black')
    axes[0].set_xlabel('Total Counts (Genes)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Library Size Distribution - Genes')
    axes[0].set_yscale('log')
    
    axes[1].hist(adata_transcripts.obs['total_counts'], bins=50, edgecolor='black')
    axes[1].set_xlabel('Total Counts (Transcripts)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Library Size Distribution - Transcripts')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/library_size_distributions.png', dpi=300)
    plt.close()
    
    # 2. Expression distribution before/after log1p
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Raw counts
    raw_genes = adata_genes.layers['counts'].data if sparse.issparse(adata_genes.layers['counts']) else adata_genes.layers['counts'].flatten()
    raw_transcripts = adata_transcripts.layers['counts'].data if sparse.issparse(adata_transcripts.layers['counts']) else adata_transcripts.layers['counts'].flatten()
    
    axes[0, 0].hist(np.log10(raw_genes[raw_genes > 0] + 1), bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('log10(Raw Count + 1)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Gene Raw Count Distribution')
    
    axes[0, 1].hist(np.log10(raw_transcripts[raw_transcripts > 0] + 1), bins=100, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('log10(Raw Count + 1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Transcript Raw Count Distribution')
    
    # Log1p normalized
    log_genes = adata_genes.layers['log1p'].data if sparse.issparse(adata_genes.layers['log1p']) else adata_genes.layers['log1p'].flatten()
    log_transcripts = adata_transcripts.layers['log1p'].data if sparse.issparse(adata_transcripts.layers['log1p']) else adata_transcripts.layers['log1p'].flatten()
    
    axes[1, 0].hist(log_genes[log_genes > 0], bins=100, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('log1p Normalized Expression')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Gene log1p Distribution')
    
    axes[1, 1].hist(log_transcripts[log_transcripts > 0], bins=100, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('log1p Normalized Expression')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Transcript log1p Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/expression_distributions.png', dpi=300)
    plt.close()
    
    # 3. Sparsity analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    gene_sparsity = (adata_genes.X == 0).sum(axis=0) / adata_genes.n_obs * 100
    transcript_sparsity = (adata_transcripts.X == 0).sum(axis=0) / adata_transcripts.n_obs * 100
    
    axes[0].hist(np.array(gene_sparsity).flatten(), bins=50, edgecolor='black')
    axes[0].set_xlabel('Sparsity (%)')
    axes[0].set_ylabel('Number of Genes')
    axes[0].set_title(f'Gene Sparsity (Mean: {np.mean(gene_sparsity):.2f}%)')
    
    axes[1].hist(np.array(transcript_sparsity).flatten(), bins=50, edgecolor='black')
    axes[1].set_xlabel('Sparsity (%)')
    axes[1].set_ylabel('Number of Transcripts')
    axes[1].set_title(f'Transcript Sparsity (Mean: {np.mean(transcript_sparsity):.2f}%)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparsity_analysis.png', dpi=300)
    plt.close()
    
    # 4. Cell type distribution
    if 'cell_type' in adata_genes.obs.columns:
        plt.figure(figsize=(10, 6))
        cell_type_counts = adata_genes.obs['cell_type'].value_counts()
        plt.barh(range(len(cell_type_counts)), cell_type_counts.values)
        plt.yticks(range(len(cell_type_counts)), cell_type_counts.index)
        plt.xlabel('Number of Samples')
        plt.ylabel('Cell Type')
        plt.title('Cell Type Distribution')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cell_type_distribution.png', dpi=300)
        plt.close()
    
    # 5. Isoform diversity per gene
    isoforms_per_gene = []
    for gene_id in adata_genes.var_names:
        transcripts = adata_genes.uns['gene_to_transcript_map'].get(gene_id, [])
        isoforms_per_gene.append(len(transcripts))
    
    plt.figure(figsize=(8, 6))
    plt.hist(isoforms_per_gene, bins=range(0, max(isoforms_per_gene)+2), edgecolor='black')
    plt.xlabel('Number of Isoforms per Gene')
    plt.ylabel('Frequency')
    plt.title(f'Isoform Diversity (Mean: {np.mean(isoforms_per_gene):.2f})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/isoform_diversity.png', dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

# Load cleaned data
adata_bulk_genes = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad')
adata_bulk_transcripts = ad.read_h5ad('/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad')

# Generate visualizations
visualize_data_distributions(adata_bulk_genes, adata_bulk_transcripts)
