from json import load
import os
# Force CUDA device 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
from pathlib import Path
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import glob

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Choice of the mode: Single-Cell ('sc') or Bulk ('bulk')
MODE = 'bulk'  

# Choice of the type: Guided (True) or Non-Guided (False)
GUIDED = False 

# Real data paths
REAL_DATA_PATHS = {
    'sc': '/work3/s193518/scIsoPred/data/sc_processed_transcripts.h5ad',    # Chemin vers votre dataset SC réel
    'bulk': '/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad'  # Chemin vers votre dataset Bulk réel
}

# ==========================================
# 2. SETUP & PATHS
# ==========================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


# local import of VAE model
try:
    from src.VAE.VAE_model import VAE
    from src.preprocessing.cell_datasets_loader import load_VAE
except ImportError:
    print("Erreur: Impossible d'importer VAE_model. Vérifiez votre sys.path.")
    sys.exit(1)

# zhome path hpc
ZHOME_PATH = "/zhome/5b/d/223428/DTU_DL_PROJECT_DIFFUSION/"

# Configuration of paths
DATA_ROOT = f'{ZHOME_PATH}data/{MODE}'
WEIGHTS_ROOT = f'{ZHOME_PATH}weights'

if GUIDED:
    INPUT_DIR = f'{DATA_ROOT}/guided'
else:
    INPUT_DIR = f'{DATA_ROOT}/non_guided'

OUTPUT_DIR = f"output/umap_{MODE}_{'guided' if GUIDED else 'non_guided'}"

# mkdir output if not exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"=== CONFIGURATION ===")
print(f"Mode: {MODE}")
print(f"Type: {'Guided' if GUIDED else 'Non-Guided'}")
print(f"Input Dir: {INPUT_DIR}")
print(f"Weights Dir: {WEIGHTS_ROOT}")
print(f"Output Dir: {OUTPUT_DIR}")
print(f"=====================")

# Style
sc.settings.set_figure_params(dpi=300, frameon=False, facecolor='white')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = False

# ==========================================
# 3. FONCTIONS
# ==========================================

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def process_and_plot(adata_combined, title, filename):
    """Pipeline Scanpy standard pour UMAP + Plot."""
    # Preprocessing spécifique pour la visualisation
    sc.pp.highly_variable_genes(adata_combined, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_combined.raw = adata_combined
    adata_combined = adata_combined[:, adata_combined.var.highly_variable]
    sc.pp.scale(adata_combined)
    sc.tl.pca(adata_combined, svd_solver='arpack')
    sc.pp.neighbors(adata_combined, n_neighbors=15, n_pcs=20)
    sc.tl.umap(adata_combined)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    custom_palette = {'Real': '#b0bec5', 'Generated': '#e91e63'} # Gris vs Rose
    
    sc.pl.umap(
        adata_combined,
        color='Condition',
        palette=custom_palette,
        alpha=0.7,
        size=20,
        ax=ax,
        show=False,
        title=title,
        frameon=False,
        legend_loc='right margin'
    )
    plt.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    device = get_device()
    print(f"Device: {device}")

    # --- A. Chargement Données Réelles ---
    real_data_path = REAL_DATA_PATHS[MODE]
    print(f"Loading Real Data from {real_data_path}...")
    if not Path(real_data_path).exists():
        raise FileNotFoundError(f"Dataset réel introuvable : {real_data_path}")
    
        
    adata_real = sc.read_h5ad(str(real_data_path), backed='r')
    # Subset pour éviter la surcharge mémoire (ex: 5000 cellules)
    adata_subset = adata_real[:5000].to_memory()
    adata_subset.var_names_make_unique()
    
    # Normalisation (essentiel pour matcher le VAE)
    sc.pp.filter_cells(adata_subset, min_genes=10)
    sc.pp.normalize_total(adata_subset, target_sum=1e4)
    sc.pp.log1p(adata_subset)
    
    real_tensor = adata_subset.X
    if hasattr(real_tensor, "toarray"):
        real_tensor = real_tensor.toarray()

    # --- B. Chargement Modèle VAE ---
    input_dim = adata_subset.shape[1]
    
    vae_path = f"{WEIGHTS_ROOT}/model_vae_{MODE}.pt"
    vae = load_VAE(
        vae_path=vae_path,
        num_gene=input_dim,
        hidden_dim=128
    )

    vae.to(device)
    vae.eval()
    print("Model loaded.")

    # --- C. Boucle de Traitement (Guided vs Non-Guided) ---
    
    if not GUIDED:
        # CAS 1: NON-GUIDED (1 seul fichier global)
        file_name = f"{MODE}_250000.npz"
        file_path = Path(INPUT_DIR) / file_name
        
        if file_path.exists():
            print(f"Processing Global file: {file_path.name}")
            npz = np.load(file_path)
            latent_gen = npz['cell_gen'][:5000] # Même nombre que réel
            
            # Decode
            with torch.no_grad():
                t_in = torch.tensor(latent_gen, dtype=torch.float32).to(device)
                decoded = vae(t_in, return_decoded=True).cpu().numpy()
            
            # Combine
            X_final = np.concatenate([real_tensor, decoded], axis=0)
            obs = ['Real'] * len(real_tensor) + ['Generated'] * len(decoded)
            adata_final = sc.AnnData(X=X_final)
            adata_final.obs['Condition'] = obs
            
            process_and_plot(adata_final, f"{MODE.upper()} Global UMAP", "UMAP_Global.png")
        else:
            print(f"Error: File {file_path} not found.")
            
    else:
        # CAS 2: GUIDED (Multiples fichiers par cluster/Leiden)
        # Recherche de tous les fichiers correspondants au pattern
        pattern = f"{MODE}_250000_leiden*.npz"
        files = sorted(list(INPUT_DIR.glob(pattern)))
        
        print(f"Found {len(files)} cluster files.")
        
        for fpath in files:
            # Extraction du numéro de cluster (ex: 'leiden12' -> '12')
            # Hypothèse nom: "sc_250000_leiden12.npz"
            try:
                cluster_id = fpath.stem.split('leiden')[-1] # Récupère la partie après 'leiden'
                print(f"--> Processing Cluster {cluster_id}")
                
                # 1. Load Gen Data
                npz = np.load(fpath)
                latent_gen = npz['cell_gen'][:2000]
                
                with torch.no_grad():
                    t_in = torch.tensor(latent_gen, dtype=torch.float32).to(device)
                    decoded = vae(t_in, return_decoded=True).cpu().numpy()
                
                # 2. Filter Real Data for this Cluster
                # On suppose que 'leiden' est une colonne dans le dataset réel
                if 'leiden' in adata_subset.obs:
                    real_mask = adata_subset.obs['leiden'] == cluster_id
                    real_cluster_data = real_tensor[real_mask]
                    
                    if len(real_cluster_data) < 5:
                        print(f"Skipping Cluster {cluster_id}: Not enough real cells.")
                        continue
                else:
                    print("Warning: 'leiden' column missing in real data.")
                    break
                
                # 3. Combine
                X_final = np.concatenate([real_cluster_data, decoded], axis=0)
                obs = ['Real'] * len(real_cluster_data) + ['Generated'] * len(decoded)
                adata_final = sc.AnnData(X=X_final)
                adata_final.obs['Condition'] = obs
                
                process_and_plot(adata_final, f"Cluster {cluster_id} ({MODE})", f"UMAP_Cluster_{cluster_id}.png")
                
            except Exception as e:
                print(f"Error on {fpath.name}: {e}")

if __name__ == "__main__":
    main()
