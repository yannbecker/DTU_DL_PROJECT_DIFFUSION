import math
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.VAE.VAE_model import VAE
from src.preprocessing.PCA import run_pca

def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data '''
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
    # Mapping location to cpu/cuda handled by torch.load generally, but rigorous mapping helps
    autoencoder.load_state_dict(torch.load(vae_path, map_location='cuda'))
    return autoencoder

def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
    use_pca=False,
    pca_dim=None, 
    plot_pca=True,
    plot_path='output/plots/pca_variance.png',
    save_pca_path='output/data/pca_reduced_data.h5ad',
    condition_key=None,  # NOUVEAU PARAMÈTRE
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.
    
    :param condition_key: Column name in adata.obs to use as label (y) for the classifier.
                          Example: 'leiden' for cell types, or 'gene_id' if rows are gene-specific.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = ad.read_h5ad(data_dir)

    # 1. Gestion des Labels (Conditionning)
    labels = None
    num_classes = 0
    print(f"Condition key: {condition_key}")
    if condition_key is not None:
        print("Entering condition key related part")
        if condition_key in adata.obs.columns:
            print(f"Loading labels from adata.obs['{condition_key}']...")
            # Encodage des labels (str -> int)
            le = LabelEncoder()
            # On convertit en string pour éviter les erreurs si mix types
            raw_labels = adata.obs[condition_key].astype(str).values 
            labels = le.fit_transform(raw_labels)
            num_classes = len(le.classes_)
            print(f"Found {num_classes} classes: {le.classes_}")
        else:
            raise KeyError(f"La clé '{condition_key}' n'existe pas dans adata.obs. Clés disponibles: {adata.obs.columns.tolist()}")

    # 2. Pré-traitement classique (Normalisation / Log1p)
    # Note: Si c'est déjà normalisé, scanpy le détectera souvent ou c'est à gérer en amont.
    # Ici on garde ta logique existante.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Data normalized and log-transformed.")
    
    cell_data = adata.X
    # Conversion sparse -> dense si nécessaire
    if hasattr(cell_data, "toarray"):
        cell_data = cell_data.toarray()
        
    print(f"Original data shape: {cell_data.shape}")

    # 3. PCA
    if use_pca:
        # Note: run_pca modifie l'adata ou renvoie numpy, assurons-nous de la cohérence
        cell_data_pca, pca_model = run_pca(adata, 
                                        n_components=None, 
                                        threshold=0.90, 
                                        plot=plot_pca, 
                                        plot_path=plot_path)
        cell_data = cell_data_pca # On remplace cell_data par la version réduite
        print(f"PCA reduced data shape: {cell_data.shape}")
        
        if save_pca_path is not None:
            # Create AnnData with PCA data and original metadata
            adata_pca = ad.AnnData(X=cell_data, obs=adata.obs.copy())
            adata_pca.write_h5ad(save_pca_path)
            print(f"Saved PCA-reduced data to {save_pca_path}")

    # 4. VAE Latent Space
    if not train_vae:
        if vae_path is None:
            raise ValueError("vae_path must be provided if train_vae is False")
            
        num_gene = cell_data.shape[1]   # now this is pca_dim or original dim
        autoencoder = load_VAE(vae_path, num_gene, hidden_dim)
        autoencoder.eval()
        
        # Passage par lot pour éviter OOM sur GPU si le dataset est énorme
        batch_size_inference = 512
        latent_list = []
        
        with torch.no_grad():
            for i in range(0, len(cell_data), batch_size_inference):
                batch = torch.tensor(cell_data[i:i+batch_size_inference]).float().cuda()
                latent = autoencoder(batch, return_latent=True)
                latent_list.append(latent.cpu().detach().numpy())
        
        cell_data = np.concatenate(latent_list, axis=0)
        print(f"VAE Latent shape: {cell_data.shape}")

    # 5. Création du Dataset et Loader
    dataset = CellDataset(
        cell_data,
        labels=labels  # On passe les labels encodés ici
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
        labels=None,
    ):
        super().__init__()
        self.data = cell_data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        
        # Si on a des labels, on les ajoute dans le dictionnaire de sortie
        # La clé "y" est celle attendue par classifier_train.py
        if self.labels is not None:
            out_dict["y"] = np.array(self.labels[idx], dtype=np.int64)
            
        return arr, out_dict
    
if __name__ == "__main__":
    print("Entering main ...")
    data_generator = load_data(
        data_dir="/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad",
        batch_size=128,
        vae_path='/zhome/70/a/224464/DL_project17/DTU_DL_PROJECT_DIFFUSION/src/VAE/output/ae_checkpoint/vae_bulk_transcript_pca/model_seed=0_step=1999.pt',
        hidden_dim=128,
        train_vae=False,
        condition_key = "leiden",
    )
    print("Done loading data")

    try:
        batch_data, extra_dict = next(data_generator)
        
        print("-" * 30)
        print("SUCCESS!")
        print(f"Batch data shape: {batch_data.shape}")
        
        if "y" in extra_dict:
            print(f"Labels shape: {extra_dict['y'].shape}")
            print(f"Example labels: {extra_dict['y'][:5]}") # Affiche les 5 premiers labels
        else:
            print("No labels found in output dictionary.")
            
    except StopIteration:
        print("The generator is empty!")
    except Exception as e:
        print(f"An error occurred: {e}")