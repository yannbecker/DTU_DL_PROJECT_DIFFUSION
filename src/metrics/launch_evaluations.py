"""
Usage:
    python evaluate_embeddings.py --real path/to/real.h5ad --generated path/to/gen.npz \
        [--emb-key X_scimilarity] [--device cpu] [--out results.json]
"""

import argparse
import json
from os import path
from pathlib import Path
import numpy as np
import anndata as ad
import torch
import time


from evaluations2 import MMD_METRICS, WASSERSTEIN_METRICS, KL_METRICS


def load_generated_npz(npz_path: str) -> torch.Tensor:
    """
    Load the first (and only) array from a .npz file and return as a torch.FloatTensor.

    Args:
        npz_path (str): Path to the .npz file.

    Returns:
        torch.Tensor: Tensor of shape (n_samples, n_features), dtype=torch.float32.
    """
    data = np.load(npz_path)
    array = list(data.values())[0]  # get the first array in the npz
    tensor = torch.from_numpy(array).float()

    print(f"Loaded array from {npz_path}")
    print(f"Shape: {tensor.shape}, dtype: {tensor.dtype}")

    return tensor



# pas bon du tout
def load_real_h5ad(h5ad_path: str, emb_key: str = "X_scimilarity") -> np.ndarray:
    adata = ad.read_h5ad(h5ad_path)
    if emb_key not in adata.obsm:
        raise KeyError(f"Embedding key '{emb_key}' not found in AnnData. Available keys: {list(adata.obsm.keys())}")
    return np.asarray(adata.obsm[emb_key])

# -------------------------
# Utility: ensure torch tensors
# -------------------------
def to_torch(x: np.ndarray, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)

# -------------------------
# Batch helpers (for memory heavy ops)
# -------------------------
def pairwise_batches(a: torch.Tensor, b: torch.Tensor, batch_size=5000):
    """
    Yield pairs (a_batch, b_batch) for processing in blocks.
    This helper isn't used for kernels that expect full matrices; it's here if needed.
    """
    n = a.shape[0]
    for i in range(0, n, batch_size):
        yield a[i : i + batch_size], b

# -------------------------
# Evaluation runner
# -------------------------
def evaluate(real_np: np.ndarray, gen_np: np.ndarray, device: str = "cpu"):
    results = {}
    # convert to torch tensors
    real = to_torch(real_np, device=device)
    gen = to_torch(gen_np, device=device)

    # dims
    results["n_real"] = int(real.shape[0])
    results["n_gen"] = int(gen.shape[0])
    results["dim"] = int(real.shape[1])

    # sanity
    if real.shape[1] != gen.shape[1]:
        raise ValueError(f"Dim mismatch: real dim {real.shape[1]} vs gen dim {gen.shape[1]}")

    # --- KL metrics (your KL objects expect torch tensors) ---
    for name, fn in KL_METRICS.items():
        start = time.time()
        try:
            val = fn().to(device)(real, gen) if hasattr(fn, "__call__") and isinstance(fn, type) else fn(real, gen)
        except TypeError:
            # some classes in your module are classes; handle both
            val = fn(real, gen)
        # val might be a tensor
        if isinstance(val, torch.Tensor):
            val = float(val.detach().cpu().item())
        results[name] = val
        results[f"{name}_time_s"] = time.time() - start

    # --- MMD metrics ---
    for name, mmd in MMD_METRICS.items():
        start = time.time()
        # mmd is an nn.Module instance in your module; ensure it's on device
        try:
            mmd = mmd.to(device)
        except Exception:
            pass
        with torch.no_grad():
            out = mmd(real, gen)
        if isinstance(out, torch.Tensor):
            out_val = float(out.detach().cpu().item())
        else:
            out_val = float(out)
        results[name] = out_val
        results[f"{name}_time_s"] = time.time() - start

    # --- Wasserstein metrics (may call POT via your wasserstein wrapper) ---
    for name, wass_fn in WASSERSTEIN_METRICS.items():
        start = time.time()
        # your wasserstein functions expect torch tensors; they return floats
        try:
            val = wass_fn(real, gen)
            # if returns torch tensor
            if isinstance(val, torch.Tensor):
                val = float(val.detach().cpu().item())
        except Exception as e:
            # fallback: try cpu numpy conversion and call with numpy-based fallback if needed
            print(f"[WARN] Wasserstein metric {name} failed with device {device}: {e}. Trying CPU numpy fallback.")
            real_cpu = real.detach().cpu().numpy()
            gen_cpu = gen.detach().cpu().numpy()
            # If your module doesn't provide numpy fallback, raise
            raise
        results[name] = val
        results[f"{name}_time_s"] = time.time() - start

    return results

# -------------------------
# CLI & main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate generated embeddings vs real embeddings using your metrics module.")
    parser.add_argument("--real", required=True, help="Path to real AnnData .h5ad file")
    parser.add_argument("--generated", required=True, help="Path to generated embeddings .npz file")
    parser.add_argument("--emb-key", default="X_scimilarity", help="AnnData.obsm key for real embeddings")
    parser.add_argument("--npz-key", default=None, help="Key inside npz for embeddings (default: try 'embeddings'/'arr_0')")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run metrics on")
    parser.add_argument("--out", default="metrics_results.json", help="Output JSON file for metrics")
    args = parser.parse_args()

    real_np = load_real_h5ad(args.real, args.emb_key)
    gen_np = load_generated_npz(args.generated)

    print(f"Loaded real {real_np.shape}, generated {gen_np.shape}")

    # convert dtype float32
    real_np = real_np.astype(np.float32)
    gen_np = gen_np.astype(np.float32)

    # trim or subsample to reasonable sizes for heavy computations? optional
    # if too large, you might want to subsample:
    max_samples = 20000
    if real_np.shape[0] > max_samples:
        idx = np.random.choice(real_np.shape[0], size=max_samples, replace=False)
        real_np = real_np[idx]
    if gen_np.shape[0] > max_samples:
        idx = np.random.choice(gen_np.shape[0], size=max_samples, replace=False)
        gen_np = gen_np[idx]

    results = evaluate(real_np, gen_np, device=args.device)

    # write results
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
