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

def load_real_data(path, num_samples=20000):
    """
    Loads real data. 
    If num_samples is -1, loads all data.
    Otherwise, loads the first num_samples.
    """
    print(f"Loading real data from {path}...")
    
    # Load in backed mode first to avoid checking file size manually
    adata_real = sc.read_h5ad(path, backed='r')
    
    # specific logic to handle "All Data" vs "Subset"
    if num_samples == -1 or num_samples >= adata_real.n_obs:
        print(f"--> Loading ALL {adata_real.n_obs} cells.")
        adata_subset = adata_real.to_memory()
    else:
        print(f"--> Loading first {num_samples} cells out of {adata_real.n_obs}.")
        # WARNING: This takes the first N cells. If your data is sorted by cluster,
        # you might miss specific clusters. Ideally, shuffle the .h5ad beforehand
        # or use random indices here if memory allows.
        adata_subset = adata_real[:num_samples].to_memory()

    adata_subset.var_names_make_unique()
    
    # Basic preprocessing
    sc.pp.filter_cells(adata_subset, min_genes=10)
    sc.pp.normalize_total(adata_subset, target_sum=1e4)
    sc.pp.log1p(adata_subset)
    
    real_tensor = adata_subset.X
    if hasattr(real_tensor, "toarray"):
        real_tensor = real_tensor.toarray()
        
    return adata_subset, real_tensor

def plot_umap(adata_combined, title, filename, color_by='Condition'):
    print(f"Generating plot: {title} (Color by: {color_by})")
    
    # Scanpy Pipeline
    # Lowering min_mean slightly to ensure enough genes are kept for clustering views
    sc.pp.highly_variable_genes(adata_combined, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_combined = adata_combined[:, adata_combined.var.highly_variable]
    
    # Scaling and PCA
    sc.pp.scale(adata_combined)
    sc.tl.pca(adata_combined, svd_solver='arpack')

    # Neighbors and UMAP
    sc.pp.neighbors(adata_combined, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_combined)
    
    fig, ax = plt.subplots(figsize=(7, 7)) # Slightly wider for legend
    
    # Logic for palette selection
    if color_by == 'Condition':
        custom_palette = {'Real': '#263238', 'Generated': '#e91e63'}
    else:
        # Use default Scanpy palette for clusters (handles many colors better)
        custom_palette = None 
    
    sc.pl.umap(
        adata_combined, 
        color=color_by, 
        palette=custom_palette, 
        alpha=0.75,
        size=25, 
        ax=ax, 
        show=False, 
        title=title, 
        frameon=False,
        legend_loc='right margin',
        legend_fontsize=12,
        legend_fontweight='bold'
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
    adata_subset, real_tensor = load_real_data(paths['real_data_path'], args.num_samples)
    
    # B. VAE Setup
    input_dim = adata_subset.shape[1]
    vae = VAE(num_genes=input_dim, device=device, seed=0, loss_ae='mse', hidden_dim=128, decoder_activation='ReLU')
    state_dict = torch.load(paths['weights_path'], map_location='cpu')
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    print("VAE loaded.")

    k_real = args.k if args.k is not None else args.num_samples
    # C. Processing
    
    # ---------------------------------------------------------
    # MODE 1: GUIDED + SPECIFIC CLUSTERS 
    # ---------------------------------------------------------
    if args.guided and args.clusters:
        print(f"--- Generating Combined Plot for Clusters: {args.clusters} ---")
        
        # Containers for concatenation
        X_list = []
        obs_leiden = []
        obs_condition = []
        
        # 1. Process Real Data for these clusters
        # Ensure strict string matching
        target_clusters = [str(c) for c in args.clusters]
        
        if 'leiden' not in adata_subset.obs:
            raise ValueError("Real data does not have 'leiden' column.")

        for cluster_id in target_clusters:
            # 1. Real Data
            mask = adata_subset.obs['leiden'] == cluster_id
            real_cluster_data = real_tensor[mask]
            
            if len(real_cluster_data) > 0:
                # Subsample real data to k
                n_real = min(len(real_cluster_data), k_real)
                idx = np.random.choice(len(real_cluster_data), n_real, replace=False)
                real_sub = real_cluster_data[idx]
                
                X_list.append(real_sub)
                obs_leiden.extend([cluster_id] * n_real)
                obs_condition.extend(['Real'] * n_real)

            # 2. Generated Data
            file_name = f"{args.mode}_250000_leiden{cluster_id}.npz"
            fpath = os.path.join(paths['input_dir'], file_name)
            
            if os.path.exists(fpath):
                npz = np.load(fpath)
                # Limit generated cells to num_samples
                latent_gen = npz['cell_gen']
                n_gen = min(len(latent_gen), args.num_samples)
                latent_sub = latent_gen[:n_gen]
                
                with torch.no_grad():
                    t_in = torch.tensor(latent_sub, dtype=torch.float32).to(device)
                    decoded = vae(t_in, return_decoded=True).cpu().numpy()
                
                X_list.append(decoded)
                obs_leiden.extend([cluster_id] * n_gen)
                obs_condition.extend(['Generated'] * n_gen)
            else:
                print(f"Warning: File not found for Cluster {cluster_id}")

        if len(X_list) > 0:
            X_final = np.concatenate(X_list, axis=0)
            adata_final = sc.AnnData(X=X_final)
            adata_final.obs['leiden'] = obs_leiden
            adata_final.obs['Condition'] = obs_condition
            adata_final.obs['leiden'] = adata_final.obs['leiden'].astype('category')
            
            out_name = f"{paths['output_dir']}/UMAP_Combined_Clusters_{'_'.join(target_clusters)}.png"
            plot_umap(adata_final, f"Combined Clusters ({args.mode.upper()})", out_name, color_by='leiden')

    # ---------------------------------------------------------
    # MODE 2: GUIDED (Iterate all individual files)
    # ---------------------------------------------------------
    elif args.guided:
        pattern = f"{paths['input_dir']}/{args.mode}_250000_leiden*.npz"
        files = sorted(glob.glob(pattern))
        print(f"Cluster files found: {len(files)}")
        
        for fpath in files:
            filename = os.path.basename(fpath)
            cluster_id = filename.split('leiden')[-1].replace('.npz', '')
            
            print(f"--> Cluster {cluster_id}")
            
            npz = np.load(fpath)
            latent_gen = npz['cell_gen']
            n_gen = min(len(latent_gen), args.num_samples)
            latent_sub = latent_gen[:n_gen]
            
            with torch.no_grad():
                t_in = torch.tensor(latent_sub, dtype=torch.float32).to(device)
                decoded = vae(t_in, return_decoded=True).cpu().numpy()
            
            if 'leiden' not in adata_subset.obs:
                break
            
            real_mask = adata_subset.obs['leiden'] == cluster_id
              
            real_cluster_data = real_tensor[real_mask]
            if len(real_cluster_data) < 5:
                continue

            n_real = min(len(real_cluster_data), k_real)
            idx = np.random.choice(len(real_cluster_data), n_real, replace=False)
            real_sub = real_cluster_data[idx]

            X_final = np.concatenate([real_sub, decoded], axis=0)

            adata_final = sc.AnnData(X=X_final)
            adata_final.obs['Condition'] = ['Real'] * n_real + ['Generated'] * n_gen
            
            out_name = f"{paths['output_dir']}/UMAP_Cluster_{cluster_id}.png"
            plot_umap(adata_final, f"Cluster {cluster_id} ({args.mode.upper()})", out_name, color_by='Condition')

    # ---------------------------------------------------------
    # MODE 3: NON-GUIDED (Global)
    # ---------------------------------------------------------
    else:
        file_name = f"{args.mode}_250000.npz"
        file_path = f"{paths['input_dir']}/{file_name}"
        
        if os.path.exists(file_path):
            npz = np.load(file_path)
            latent_gen = npz['cell_gen'][:len(real_tensor)]
            
            with torch.no_grad():
                t_in = torch.tensor(latent_gen, dtype=torch.float32).to(device)
                decoded = vae(t_in, return_decoded=True).cpu().numpy()
            
            X_final = np.concatenate([real_tensor, decoded], axis=0)
            obs = ['Real'] * len(real_tensor) + ['Generated'] * len(decoded)
            adata_final = sc.AnnData(X=X_final)
            adata_final.obs['Condition'] = obs
            
            out_name = f"{paths['output_dir']}/UMAP_Global.png"
            plot_umap(adata_final, f"{args.mode.upper()} Global UMAP", out_name, color_by='Condition')
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['sc', 'bulk'], required=True)
    parser.add_argument('--guided', action='store_true')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--clusters', nargs='+', type=str, default=None, help="List of specific cluster IDs to combine (e.g. 0 1 4)")
    parser.add_argument('--num_samples', type=int, default=1000, 
                        help="Number of generated samples to use per cluster (or global). Default 1000.")
    
    parser.add_argument('--k', type=int, default=None,
                        help="Max number of real samples per cluster. Defaults to num_samples if not set.")
    
    parser.add_argument('--num_real_load', type=int, default=-1, 
                        help="Total number of real cells to load from disk (-1 for all). Default -1.")
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
