import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

def load_data(*, data_dir, batch_size, deterministic=False, condition_key=None):
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = ad.read_h5ad(data_dir)

    labels = None
    if condition_key is not None:
        if condition_key in adata.obs.columns:
            le = LabelEncoder()
            raw_labels = adata.obs[condition_key].astype(str).values
            labels = le.fit_transform(raw_labels)
            print(f"Found {len(le.classes_)} classes: {le.classes_}")
        else:
            raise KeyError(f"Condition key '{condition_key}' not in adata.obs. Available keys: {adata.obs.columns.tolist()}")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_data = adata.X
    if hasattr(cell_data, "toarray"):
        cell_data = cell_data.toarray()

    dataset = CellDataset(cell_data, labels=labels)

    if deterministic:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    while True:
        yield from loader

class CellDataset(Dataset):
    def __init__(self, cell_data, labels=None):
        super().__init__()
        self.data = cell_data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.labels is not None:
            out_dict["y"] = np.array(self.labels[idx], dtype=np.int64)
        return arr, out_dict

if __name__ == "__main__":
    print("Entering main ...")
    data_generator = load_data(
        data_dir="/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad",
        batch_size=128,
        condition_key="leiden",
    )
    print("Done loading data")

    try:
        batch_data, extra_dict = next(data_generator)

        print("-" * 30)
        print("SUCCESS!")
        print(f"Batch data shape: {batch_data.shape}")

        if "y" in extra_dict:
            print(f"Labels shape: {extra_dict['y'].shape}")
            print(f"Example labels: {extra_dict['y'][:5]}")
        else:
            print("No labels found in output dictionary.")

    except StopIteration:
        print("The generator is empty!")
    except Exception as e:
        print(f"An error occurred: {e}")
