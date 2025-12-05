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
    from evaluations2 import compute_kl, compute_mmd, compute_wasserstein
except ImportError:
    print(f"Error: Unable to import VAE_model from path: {HPC_ROOT}")
    sys.exit(1)

# ==========================================
# 2. METRICS FUNCTIONS
# ==========================================


def compute_correlations():
    pass  # A completer

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
    """Computes all metrics for a given pair of real and generated data."""
    # Ensure sizes match by subsampling the larger set
    n_samples = min(len(real_data), len(gen_data))
    idx_real = np.random.choice(len(real_data), n_samples, replace=False)
    idx_gen = np.random.choice(len(gen_data), n_samples, replace=False)
    real_sub, gen_sub = real_data[idx_real], gen_data[idx_gen]

    # PCA for distribution metrics (MMD, Wasserstein)
    pca = PCA(n_components=30)
    combined = np.concatenate([real_sub, gen_sub], axis=0)
    pca.fit(combined)
    real_pca, gen_pca = pca.transform(real_sub), pca.transform(gen_sub)

    # Calculate all metrics
    mmd_val = compute_mmd(real_pca, gen_pca)
    w_val = compute_wasserstein(real_pca, gen_pca)
    corr_mean, corr_var = compute_correlations(real_sub, gen_sub)
    
    return {
        "Condition": label, "MMD": mmd_val, "Wasserstein": w_val,
        "Corr_Mean": corr_mean, "Corr_Var": corr_var, "N_Samples": n_samples
    }

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

    # C. Process Generated Data Files
    ...  # A completer
    results_list = []

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

        