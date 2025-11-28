"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    Modifié pour forcer un environnement mono-GPU/mono-processus DDP.
    """
    if dist.is_initialized():
        return
    
    # Forcez la carte graphique visible si nécessaire (déjà fait, mais gardons-le)
    # Note: L'environnement externe 'CUDA_VISIBLE_DEVICES=3' devrait prévaloir
    # sur cette ligne si elle est présente.
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    # --- SIMULATION DES VARIABLES D'ENVIRONNEMENT MPI/DISTRIBUÉES ---
    # Pour que PyTorch DDP puisse s'initialiser correctement en mode mono-processus.
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    
    # Chercher un port libre
    try:
        port = _find_free_port()
    except:
        # Fallback
        port = 29500 
        
    os.environ["MASTER_PORT"] = str(port)
    
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    
    # Initialise le groupe de processus PyTorch
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")

# Modified function
def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file. Dans un environnement mono-GPU, on charge simplement.
    """
    if dist.get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)
    else:
        raise RuntimeError("Non-rank 0 reached load_state_dict in single-process mode.")


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
