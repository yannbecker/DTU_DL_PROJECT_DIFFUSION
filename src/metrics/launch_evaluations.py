"""
Usage:
    python evaluate_embeddings.py --real path/to/real.h5ad --generated path/to/gen.npz \
        [--device cpu] [--out results.json] [--max-cells 20000]
"""

import argparse
import json
from pathlib import Path
import numpy as np
import anndata as ad
import scanpy as sc
import torch
import time

from evaluations2 import MMD_METRICS, WASSERSTEIN_METRICS, KL_METRICS
from VAE_model import VAE


# -------------------------
# Load VAE
# -------------------------
def load_VAE(device: str = "cpu") -> torch.nn.Module:
    autoencoder = VAE(
        num_genes=162009,
        device=device,
        seed=0,
        loss_ae='mse',
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(
        '/model_seed=0_step=1999.pt',
        map_location=device
    ))
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


# -------------------------
# Load real features
# -------------------------
# -------------------------
# Load real features
# -------------------------
def load_real_features(h5ad_path: str) -> np.ndarray:
    adata = ad.read_h5ad(h5ad_path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    return X.astype(np.float32)



# -------------------------
# Load generated embeddings and decode via VAE
# -------------------------
def load_generated_features(npz_path: str, vae: torch.nn.Module, device: str = "cpu", max_cells: int = None) -> np.ndarray:
    npzfile = np.load(npz_path, allow_pickle=True)
    latent = npzfile['cell_gen']
    if max_cells is not None:
        latent = latent[:max_cells]

    latent = torch.tensor(latent).float().to(device)
    vae.to(device)
    with torch.no_grad():
        decoded = vae(latent, return_decoded=True).detach().cpu().numpy()

    return decoded.astype(np.float32)


# -------------------------
# Utility: torch conversion
# -------------------------
def to_torch(x: np.ndarray, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)


# -------------------------
# Evaluation runner
# -------------------------
def evaluate(real_np: np.ndarray, gen_np: np.ndarray, device: str = "cpu"):
    results = {}
    real = to_torch(real_np, device=device)
    gen = to_torch(gen_np, device=device)

    results["n_real"] = int(real.shape[0])
    results["n_gen"] = int(gen.shape[0])
    results["dim"] = int(real.shape[1])

    if real.shape[1] != gen.shape[1]:
        raise ValueError(f"Dim mismatch: real dim {real.shape[1]} vs gen dim {gen.shape[1]}")

    # KL metrics
    for name, fn in KL_METRICS.items():
        start = time.time()
        try:
            val = fn().to(device)(real, gen) if hasattr(fn, "__call__") and isinstance(fn, type) else fn(real, gen)
        except TypeError:
            val = fn(real, gen)
        if isinstance(val, torch.Tensor):
            val = float(val.detach().cpu().item())
        results[name] = val
        results[f"{name}_time_s"] = time.time() - start

    # MMD metrics
    for name, mmd in MMD_METRICS.items():
        start = time.time()
        try:
            mmd = mmd.to(device)
        except Exception:
            pass
        with torch.no_grad():
            out = mmd(real, gen)
        results[name] = float(out.detach().cpu().item()) if isinstance(out, torch.Tensor) else float(out)
        results[f"{name}_time_s"] = time.time() - start

    # Wasserstein metrics
    for name, wass_fn in WASSERSTEIN_METRICS.items():
        start = time.time()
        try:
            val = wass_fn(real, gen)
            if isinstance(val, torch.Tensor):
                val = float(val.detach().cpu().item())
        except Exception as e:
            print(f"[WARN] Wasserstein metric {name} failed on device {device}: {e}")
            real_cpu = real.detach().cpu().numpy()
            gen_cpu = gen.detach().cpu().numpy()
            raise
        results[name] = val
        results[f"{name}_time_s"] = time.time() - start

    return results


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate generated embeddings vs real embeddings in feature space.")
    parser.add_argument("--real", required=True, help="Path to real AnnData .h5ad file")
    parser.add_argument("--generated", required=True, help="Path to generated embeddings .npz file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run metrics on")
    parser.add_argument("--out", default="metrics_results.json", help="Output JSON file for metrics")
    parser.add_argument("--max-cells", type=int, default=20000, help="Maximum number of cells to use")
    args = parser.parse_args()

    print(f"Loading VAE on device {args.device}...")
    vae = load_VAE(device=args.device)

    print("Loading real data...")
    real_np = load_real_features(args.real)

    print("Loading generated embeddings and decoding via VAE...")
    gen_np = load_generated_features(args.generated, vae, device=args.device, max_cells=args.max_cells)

    print(f"Real shape: {real_np.shape}, Generated decoded shape: {gen_np.shape}")

    results = evaluate(real_np, gen_np, device=args.device)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
