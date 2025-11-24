import math
import random

import anndata as ad

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.VAE.VAE_model import VAE
from src.preprocessing.PCA import run_pca
from sklearn.preprocessing import LabelEncoder

def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data
    See https://f1000research.com/posters/4-1041 for motivation.
    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def load_VAE(vae_path, num_gene, hidden_dim):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder


def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
    use_pca=True,
    pca_dim=None, 
    plot_pca=True,
    plot_path='output/plots/pca_variance.png',
    save_pca_path='output/data/pca_reduced_data.h5ad'
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = ad.read_h5ad(data_dir)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Data normalized and log-transformed.")
    cell_data = adata.X.toarray()
    print(f"Original data shape: {cell_data.shape}")
    if use_pca:
        cell_data, pca_model = run_pca(adata, 
                                        n_components=None, 
                                        threshold=0.90, 
                                        plot=True, 
                                        plot_path=plot_path)
        print(f"PCA reduced data shape: {cell_data.shape}")
        print(f"PCA model components shape: {pca_model.components_.shape}")
        if save_pca_path is not None:
            # Create AnnData with PCA data and original metadata
            adata_pca = ad.AnnData(X=cell_data, obs=adata.obs.copy())
            # Optionally copy var metadata or create dummy var for PCs
            adata_pca.var['PCs'] = [f'PC{i+1}' for i in range(cell_data.shape[1])]

            adata_pca.write_h5ad(save_pca_path)
            print(f"Saved PCA-reduced data to {save_pca_path}")

    # turn data into VAE latent if not training the VAE itself
    if not train_vae:
        num_gene = cell_data.shape[1]   # now this is pca_dim
        autoencoder = load_VAE(vae_path, num_gene, hidden_dim)
        autoencoder.eval()
        with torch.no_grad():
            cell_data = autoencoder(
                torch.tensor(cell_data).cuda(),
                return_latent=True,
            )
            cell_data = cell_data.cpu().detach().numpy()
    
    dataset = CellDataset(
        cell_data
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name=None,
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        return arr, out_dict

