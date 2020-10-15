import pickle
import gzip
from typing import Tuple, Optional
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.descriptors import ecfp


def load_mnist_some_classes(include_labels: Optional[Tuple] = None, n_rows: int = None) -> np.array:
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_set, _, _ = pickle.load(f, encoding='latin1')
    train_data, train_labels = train_set
    if include_labels is None:
        include_labels = tuple(range(10))
    keep_indexes = np.in1d(train_labels, include_labels)
    train_data = train_data[keep_indexes]
    train_labels = train_labels[keep_indexes]
    if n_rows is None or n_rows > train_labels.shape[0]:
        n_rows = train_labels.shape[0]

    return torch.tensor(train_data[:n_rows]), torch.tensor(train_labels[:n_rows])


class SmilesDataset(Dataset):

    def __init__(self, filepath, r=3, n_bits=2048):
        self.filepath = filepath
        self.r = r
        self.n_bits = n_bits
        with open(self.filepath) as _file:
            self.smiles = [s.rstrip() for s in _file.readlines()]

        # Заглушка для labels
        self.labels = torch.tensor([1 for _ in self.smiles])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return ecfp(self.smiles[idx], r=self.r, nBits=self.n_bits), self.labels[idx]
