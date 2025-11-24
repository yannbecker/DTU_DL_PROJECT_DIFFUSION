import argparse
import os
import re
import time
import scanpy as sc

import numpy as np
import torch
from VAE_model import VAE
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing.cell_datasets_loader import load_data
from src.preprocessing.PCA import run_pca

torch.autograd.set_detect_anomaly(True)
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_vae(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_data(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        train_vae=True,
        use_pca=True,          
        pca_dim=args["num_genes"],  # num_genes now means PCA dim
        plot_pca=args["plot_pca"]
    )

    autoencoder = VAE(
        num_genes=args["num_genes"],
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
    )
    if state_dict is not None:
        print('loading pretrained model from: \n',state_dict)
        use_gpu = device == "cuda"
        autoencoder.encoder.load_state(state_dict["encoder"], use_gpu)
        autoencoder.decoder.load_state(state_dict["decoder"], use_gpu)

    print('autoencoder prepared')

    return autoencoder, datasets

def train_vae(args, return_model=False):
    """
    Trains a autoencoder
    """
    if args["state_dict"] is not None:
        filenames = {}
        checkpoint_path = {
            "encoder": os.path.join(
                args["state_dict"], filenames.get("model", "encoder.ckpt")
            ),
            "decoder": os.path.join(
                args["state_dict"], filenames.get("model", "decoder.ckpt")
            ),
            "gene_order": os.path.join(
                args["state_dict"], filenames.get("gene_order", "gene_order.tsv")
            ),
        }
        autoencoder, datasets = prepare_vae(args, checkpoint_path)
    else:
        autoencoder, datasets = prepare_vae(args)
   
    args["hparams"] = autoencoder.hparams

    start_time = time.time()
    for step in range(args["max_steps"]):

        genes, _ = next(datasets)

        minibatch_training_stats = autoencoder.train_step(genes)

        if step % 1000 == 0:
            for key, val in minibatch_training_stats.items():
                print('step ', step, 'loss ', val)

        ellapsed_minutes = (time.time() - start_time) / 60

        stop = ellapsed_minutes > args["max_minutes"] or (
            step == args["max_steps"] - 1
        )

        if ((step % args["checkpoint_freq"]) == 0 or stop):

            os.makedirs(args["save_dir"],exist_ok=True)
            torch.save(
                autoencoder.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_step={}.pt".format(args["seed"], step),
                ),
            )

            if stop:
                break

    if return_model:
        return autoencoder, datasets


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description="Finetune Scimilarity")
    # dataset arguments
    parser.add_argument("--data_dir", type=str, default='/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad')
    parser.add_argument("--loss_ae", type=str, default="mse")
    parser.add_argument("--decoder_activation", type=str, default="ReLU")
    parser.add_argument("--plot_pca", type=bool, default=True)

    # AE arguments                                             
    parser.add_argument("--local_rank", type=int, default=0)  
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--num_genes", type=int, default=None) # if use PCA, num_genes means PCA dim
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hparams", type=str, default="")

    # training arguments
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--checkpoint_freq", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--state_dict", type=str, default="/data1/lep/Workspace/guided-diffusion/scimilarity-main/models/annotation_model_v1")  # if pretrain
    # parser.add_argument("--state_dict", type=str, default=None)   # if not pretrain

    parser.add_argument("--save_dir", type=str, default='../output/ae_checkpoint/vae_bulk_transcript_pca/')
    parser.add_argument("--sweep_seeds", type=int, default=200)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    seed_everything(1234)
    args = parse_arguments()
    if args["num_genes"] is None:
        print("num_genes not provided, automatically detecting via PCA threshold ...")
        adata = sc.read_h5ad(args["data_dir"])
        
        # Run PCA with n_components large enough to capture all variance
        # but do not plot here to save time
        X_pca, pca = run_pca(adata, n_components=None, threshold=0.90, plot=False)
        
        # Number of components chosen by run_pca to reach threshold variance
        num_components = X_pca.shape[1]
        args["num_genes"] = num_components
        
        print(f"Automatically set num_genes to {num_components} based on 90% variance threshold.")
    autoencoder, datasets = prepare_vae(args)
    print('data loaded from ', args["data_dir"])
    print('PCA resulting data shape: ', datasets[0][0].shape)
    train_vae(parse_arguments())
