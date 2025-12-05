import os
from turtle import st

from src.utils.fp16_util import state_dict_to_master_params
# Force CUDA device 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import glob
from scipy.stats import wasserstein_distance, pearsonr
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA

# ==========================================
# 1. SETUP (Paths & Imports)
# ==========================================
# To adapt to HPC 
HPC_ROOT = "/zhome/5b/d/223428/DTU_DL_PROJECT_DIFFUSION"
sys.path.append(HPC_ROOT)

try:
    from src.VAE.VAE_model import VAE
    from src.metrics.evaluations2 import MMDLoss, wasserstein, GaussianKLDivergence

except ImportError:
    print(f"Error: Unable to import VAE_model from path: {HPC_ROOT}")
    sys.exit(1)

# ==========================================
# 2. METRICS FUNCTIONS
# ==========================================

def compute_mmd(x, y, gamma=1.0):
    x_kernel = rbf_kernel(x, x, gamma=gamma)
    y_kernel = rbf_kernel(y, y, gamma=gamma)
    xy_kernel = rbf_kernel(x, y, gamma=gamma)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

def compute_wasserstein(x, y):
    wd_list = [wasserstein_distance(x[:, i], y[:, i]) for i in range(x.shape[1])]
    return float(np.mean(wd_list))

def compute_correlations(real_data, gen_data):
    mean_real = np.mean(real_data, axis=0)
    mean_gen = np.mean(gen_data, axis=0)
    var_real = np.var(real_data, axis=0)
    var_gen = np.var(gen_data, axis=0)
    # Replace NaNs (e.g. zero-variance genes) to avoid crashes
    mean_real = np.nan_to_num(mean_real)
    mean_gen = np.nan_to_num(mean_gen)
    var_real = np.nan_to_num(var_real)
    var_gen = np.nan_to_num(var_gen)
    corr_mean, _ = pearsonr(mean_real, mean_gen)
    corr_var, _ = pearsonr(var_real, var_gen)
    return float(corr_mean), float(corr_var)


# Eventuellement faire les appels aux fonctions de metrics ici

# ==========================================
# 3. HELPER FUNCTIONS
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

def get_metrics_for_batch(real_data, gen_data, label):
    """Compute all metrics for a given pair of real and generated matrices."""
    n_samples = min(len(real_data), len(gen_data))
    if n_samples < 5:
        return None

    idx_real = np.random.choice(len(real_data), n_samples, replace=False)
    idx_gen = np.random.choice(len(gen_data), n_samples, replace=False)
    real_sub = real_data[idx_real]
    gen_sub = gen_data[idx_gen]

    # PCA on concatenated data, then evaluate metrics in PC space
    pca = PCA(n_components=min(30, real_sub.shape[1]))
    combined = np.concatenate([real_sub, gen_sub], axis=0)
    pca.fit(combined)
    real_pca = pca.transform(real_sub)
    gen_pca = pca.transform(gen_sub)

    mmd_val = compute_mmd(real_pca, gen_pca)
    w_val = compute_wasserstein(real_pca, gen_pca)
    corr_mean, corr_var = compute_correlations(real_sub, gen_sub)

    return {
        "Condition": label,
        "MMD": mmd_val,
        "Wasserstein": w_val,
        "Corr_Mean": corr_mean,
        "Corr_Var": corr_var,
        "N_Samples": int(n_samples),
    }

# def get_metrics_for_batch(real_data, gen_data, label):
#     """Computes all metrics for a given pair of real and generated data."""
#     # Ensure sizes match by subsampling the larger set
#     n_samples = min(len(real_data), len(gen_data))
#     idx_real = np.random.choice(len(real_data), n_samples, replace=False)
#     idx_gen = np.random.choice(len(gen_data), n_samples, replace=False)
#     real_sub, gen_sub = real_data[idx_real], gen_data[idx_gen]

#     # PCA already done when computing metrics 
    
#     # Calculate all metrics
#     mmd_val = compute_mmd(real_sub, gen_sub)
#     w_val = compute_wasserstein(real_sub, gen_sub)
#     corr_mean, corr_var = compute_correlations(real_sub, gen_sub)
    
#     return {
#         "Condition": label, "MMD": mmd_val, "Wasserstein": w_val,
#         "Corr_Mean": corr_mean, "Corr_Var": corr_var, "N_Samples": n_samples
#     }

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

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def run_metrics(args, paths):
    device = get_device()
    print(f"Device: {device}")
    results_list = []

    # A. Load Real Data
    adata_real, real_tensor = load_real_data(paths["real_data_path"], num_samples=5000) # num_samples to put as arg parse later
    print("Real data loaded.")

    # B. Load VAE Model
    input_dim = adata_real.shape[1]
    vae = VAE(num_genes=input_dim, device=device, seed=0, loss_ae='mse', hidden_dim=128, decoder_activation='ReLU')
    
    state_dict = torch.load(paths["weights_path"], map_location=device)
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    print("VAE loaded.")

    # Max real per cluster; if k is None, default to num_samples
    max_real_per_cluster = args.k if args.k is not None else args.num_samples

    # C. Process Generated Data Files
    if not args.guided:
        # Non-Guided: Single global comparison
        file_path = f"{paths['input_dir']}/{args.mode}_250000.npz"
        if os.path.exists(file_path):
            print("Processing Global Non-Guided...")
            latent_gen = np.load(file_path)['cell_gen']

            # Limit generated samples globally by num_samples (if provided)
            n_gen = min(len(latent_gen), args.num_samples)
            idx_gen = np.random.choice(len(latent_gen), n_gen, replace=False)
            latent_sub = latent_gen[idx_gen]

            with torch.no_grad():
                decoded_gen = vae(torch.tensor(latent_sub, dtype=torch.float32).to(device), return_decoded=True).cpu().numpy()

            res = get_metrics_for_batch(real_tensor, decoded_gen, "Global")
            if res is not None: 
                results_list.append(res)

    else:
        # Guided Mode
        files = sorted(glob.glob(f"{paths['input_dir']}/{args.mode}_250000_leiden*.npz"))
        print(f"Guided mode: {len(files)} cluster files found.")

        # --- 1. Global Guided Metric Calculation ---
        print("\nAggregating data for 'Global Guided' metric...")
        all_real, all_gen = [], []
        
        for fpath in files:
            cluster_id = os.path.basename(fpath).split('leiden')[-1].replace('.npz', '')
            if 'leiden' not in adata_real.obs:
                print("Column 'leiden' not found in real data; cannot condition by cluster.")
                break

            mask = adata_real.obs['leiden'] == cluster_id
            if mask.sum() == 0:
                continue

            real_cluster = real_tensor[mask]
            if len(real_cluster) == 0:
                continue

            npz = np.load(fpath)
            latent_gen = npz['cell_gen']

            # Limit generated samples per cluster
            n_gen = min(len(latent_gen), args.num_samples)
            if n_gen < 1:
                continue
            idx_gen = np.random.choice(len(latent_gen), n_gen, replace=False)
            latent_sub = latent_gen[idx_gen]

            # Limit real samples per cluster
            n_real = min(len(real_cluster), max_real_per_cluster)
            if n_real < 1:
                continue
            idx_real = np.random.choice(len(real_cluster), n_real, replace=False)
            real_sub = real_cluster[idx_real]

            with torch.no_grad():
                decoded_sub = vae(
                    torch.tensor(latent_sub, dtype=torch.float32).to(device),
                    return_decoded=True
                ).cpu().numpy()

            all_real.append(real_sub)
            all_gen.append(decoded_sub)

        if all_real and all_gen:
            real_global = np.concatenate(all_real, axis=0)
            gen_global = np.concatenate(all_gen, axis=0)
            res = get_metrics_for_batch(real_global, gen_global, "Guided_Global")
            if res is not None:
                results_list.append(res)
        else:
            print("No valid clusters found for Guided_Global metrics.")


        # --- 2. Per-Cluster Metrics (Optional) ---
        if args.per_cluster:
            print("\nCalculating per-cluster metrics...")
            for fpath in files:
                cluster_id = os.path.basename(fpath).split('leiden')[-1].replace('.npz', '')
                print(f"--> Cluster {cluster_id}")
                
                mask = adata_real.obs['leiden'] == cluster_id
                if mask.sum() == 0:
                    continue
                real_cluster = real_tensor[mask]
                if len(real_cluster) < 5:
                    continue

                npz = np.load(fpath)
                latent_gen = npz['cell_gen']

                # Limit generated samples per cluster
                n_gen = min(len(latent_gen), args.num_samples)
                if n_gen < 5:
                    continue
                idx_gen = np.random.choice(len(latent_gen), n_gen, replace=False)
                latent_sub = latent_gen[idx_gen]

                # Limit real samples per cluster by k/num_samples
                n_real = min(len(real_cluster), max_real_per_cluster)
                if n_real < 5:
                    continue
                idx_real = np.random.choice(len(real_cluster), n_real, replace=False)
                real_sub = real_cluster[idx_real]

                with torch.no_grad():
                    decoded_cluster = vae(
                        torch.tensor(latent_sub, dtype=torch.float32).to(device),
                        return_decoded=True
                    ).cpu().numpy()

                label = f"Cluster_{cluster_id}"
                res = get_metrics_for_batch(real_sub, decoded_cluster, label)
                if res is not None:
                    results_list.append(res)

    # D. Save Metrics Results
    if results_list:
        df = pd.DataFrame(results_list)
        out_path = f"{paths['output_dir']}/{paths['csv_name']}.csv"
        df.to_csv(out_path, index=False)
        print(f"Metrics results saved to {out_path}")
        print(df.head())
    else:
        print("No results to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Metrics on Generated Data")
    parser.add_argument('--mode', type=str, choices=['sc', 'bulk'], required=True, help='Data modality: sc or bulk')
    parser.add_argument('--guided', action='store_true', help='Whether to use guided generation')
    parser.add_argument('--transfer', action='store_true', help='Whether transfer learning was used')
    parser.add_argument('--per_cluster', action='store_true', help="In guided mode, also calculate metrics for each cluster individually.")
    parser.add_argument('--k', type=int, default=None, help="Max number of real samples per cluster in guided mode; defaults to num_samples if not set.")
    parser.add_argument('--num_real', type=int, default=5000, help="Number of real cells to load from the .h5ad file (global cap).")
    # A completer: autres arguments nécessaires

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
        run_metrics(args, paths)
    except Exception as e:
        print(f"Error during execution: {e}")

        