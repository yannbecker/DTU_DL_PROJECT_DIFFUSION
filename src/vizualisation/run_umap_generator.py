import os
# Force CUDA device 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import glob

# ==========================================
# 1. SETUP (Paths & Imports)
# ==========================================

# Manual definition of the root path to avoid Path errors. Need to use pwd in the HPC.
HPC_ROOT = "/zhome/5b/d/223428/DTU_DL_PROJECT_DIFFUSION"
sys.path.append(HPC_ROOT)

try:
    from src.VAE.VAE_model import VAE
except ImportError:
    print(f"Error: Unable to import VAE_model from path: {HPC_ROOT}")
    sys.exit(1)

# Style for plots
sc.settings.set_figure_params(dpi=300, frameon=False, facecolor='white')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = False

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def setup_paths(args):
    """Dynamically constructs paths using f-strings."""
    
    # 1. Choose DATA folder
    if args.mode == 'sc':
        if args.transfer:
            data_folder = "sc_transfer"
        else:
            data_folder = "sc"
    else:
        # Bulk mode (usually no transfer, but keeping logic clean)
        data_folder = "bulk"
    
    # 2. Choose sub-folder (guided/non_guided)
    if args.guided:
        sub_folder = "guided"
    else:
        sub_folder = "non_guided"

    # 3. Choose VAE suffix (sc or bulk)
    # The VAE depends on data modality, not transfer learning
    vae_mode = "sc" if args.mode == "sc" else "bulk"

    # 4. Construct full paths with f-strings
    data_root = f"{HPC_ROOT}/data/{data_folder}"
    input_dir = f"{data_root}/{sub_folder}"
    weights_path = f"{HPC_ROOT}/weights/model_vae_{vae_mode}.pt"
    real_data_path = args.real_data_paths[args.mode]
    
    output_folder_name = f"umap_{data_folder}_{sub_folder}"
    output_dir = f"{HPC_ROOT}/output/{output_folder_name}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "data_root": data_folder,
        "input_dir": input_dir,
        "weights_path": weights_path,
        "real_data_path": real_data_path,
        "output_dir": output_dir
    }

    # Basic checks
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")
    if not os.path.exists(real_data_path):
        raise FileNotFoundError(f"Real data file does not exist: {real_data_path}")

    return paths

def load_real_data(path, num_samples=5000):
    print(f"Loading real data from {path}...")
    adata_real = sc.read_h5ad(path, backed='r')
    adata_subset = adata_real[:num_samples].to_memory()
    adata_subset.var_names_make_unique()
    
    sc.pp.filter_cells(adata_subset, min_genes=10)
    sc.pp.normalize_total(adata_subset, target_sum=1e4)
    sc.pp.log1p(adata_subset)
    
    real_tensor = adata_subset.X
    if hasattr(real_tensor, "toarray"):
        real_tensor = real_tensor.toarray()
        
    return adata_subset, real_tensor

def plot_umap(adata_combined, title, filename):
    print(f"Generating plot: {title}")
    
    # Scanpy Pipeline
    sc.pp.highly_variable_genes(adata_combined, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_combined = adata_combined[:, adata_combined.var.highly_variable]
    sc.pp.scale(adata_combined)
    sc.tl.pca(adata_combined, svd_solver='arpack')
    sc.pp.neighbors(adata_combined, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_combined)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    custom_palette = {'Real': '#b0bec5', 'Generated': '#e91e63'}
    
    sc.pl.umap(
        adata_combined, color='Condition', palette=custom_palette, alpha=0.7,
        size=20, ax=ax, show=False, title=title, frameon=False,
        legend_loc='right margin'
    )
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved as: {filename}")

# ==========================================
# 3. MAIN LOGIC
# ==========================================

def run_visualization(args, paths):
    device = get_device()
    print(f"Device: {device}")

    # A. Real Data
    adata_subset, real_tensor = load_real_data(paths['real_data_path'])
    
    # B. VAE
    input_dim = adata_subset.shape[1]
    vae = VAE(num_genes=input_dim, device=device, seed=0, loss_ae='mse', hidden_dim=128, decoder_activation='ReLU')
    
    # Load weights on CPU first for safety
    state_dict = torch.load(paths['weights_path'], map_location='cpu')
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    print("VAE loaded.")

    # C. Processing
    if not args.guided:
        # Non-Guided Mode
        file_name = f"{args.mode}_250000.npz"
        file_path = f"{paths['input_dir']}/{file_name}"
        
        if os.path.exists(file_path):
            npz = np.load(file_path)
            # Limit to the number of real cells for visual balance
            latent_gen = npz['cell_gen'][:len(real_tensor)]
            
            with torch.no_grad():
                t_in = torch.tensor(latent_gen, dtype=torch.float32).to(device)
                decoded = vae(t_in, return_decoded=True).cpu().numpy()
            
            X_final = np.concatenate([real_tensor, decoded], axis=0)
            obs = ['Real'] * len(real_tensor) + ['Generated'] * len(decoded)
            adata_final = sc.AnnData(X=X_final)
            adata_final.obs['Condition'] = obs
            
            out_name = f"{paths['output_dir']}/UMAP_Global.png"
            plot_umap(adata_final, f"{args.mode.upper()} Global UMAP", out_name)
        else:
            print(f"File not found: {file_path}")

    else:
        # Guided Mode
        pattern = f"{paths['input_dir']}/{args.mode}_250000_leiden*.npz"
        files = sorted(glob.glob(pattern))
        print(f"Cluster files found: {len(files)}")
        
        for fpath in files:
            # Extract cluster ID from filename
            filename = os.path.basename(fpath)
            cluster_id = filename.split('leiden')[-1].replace('.npz', '')
            
            print(f"--> Cluster {cluster_id}")
            
            npz = np.load(fpath)
            latent_gen = npz['cell_gen'][:2000]
            
            with torch.no_grad():
                t_in = torch.tensor(latent_gen, dtype=torch.float32).to(device)
                decoded = vae(t_in, return_decoded=True).cpu().numpy()
            
            if 'leiden' not in adata_subset.obs:
                print("No 'leiden' column in real data.")
                break
            
            real_mask = adata_subset.obs['leiden'] == cluster_id
            if sum(real_mask) < 5:
                continue
                
            real_cluster_data = real_tensor[real_mask]
            X_final = np.concatenate([real_cluster_data, decoded], axis=0)
            adata_final = sc.AnnData(X=X_final)
            adata_final.obs['Condition'] = ['Real'] * len(real_cluster_data) + ['Generated'] * len(decoded)
            
            out_name = f"{paths['output_dir']}/UMAP_Cluster_{cluster_id}.png"
            plot_umap(adata_final, f"Cluster {cluster_id} ({args.mode.upper()})", out_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['sc', 'bulk'], required=True)
    parser.add_argument('--guided', action='store_true')
    parser.add_argument('--transfer', action='store_true')
    
    # Hardcoded paths here to avoid argument issues
    real_paths = {
        'sc': '/work3/s193518/scIsoPred/data/sc_processed_transcripts.h5ad',
        'bulk': '/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad'
    }
    
    args = parser.parse_args()
    args.real_data_paths = real_paths
    
    try:
        paths = setup_paths(args)
        print(f"--- Configuration ---")
        print(f"Input: {paths['input_dir']}")
        print(f"Output: {paths['output_dir']}")
        print(f"---------------------")
        run_visualization(args, paths)
    except Exception as e:
        print(f"Critical Error: {e}")
