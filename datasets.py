import pickle
import gzip
from typing import Tuple, Optional, Dict, Any
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.descriptors import ecfp
from utils.reactions import reaction_fps


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

    def __init__(self, filepath, dev, r=3, n_bits=2048):
        self.filepath = filepath
        self.r = r
        self.dev = dev
        self.n_bits = n_bits
        with open(self.filepath) as _file:
            self.data = [s.rstrip().split() for s in _file.readlines()]
            self.smiles = [d[0] for d in self.data]
            try:
                self.labels = [int(d[1]) for d in self.data]
            except IndexError:
                self.labels = [0 for _ in self.data]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        descriptors = torch.from_numpy(ecfp(self.smiles[idx], r=self.r, nBits=self.n_bits)).to(self.dev)
        return descriptors, self.labels[idx]


class ReactionSmilesDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 dev: Any,
                 fp_method: str,
                 params: Dict[str, Any]) -> None:
        self.filepath = filepath  # path to .csv file
        self.dev = dev
        self.smiles = []
        self.fp_method = fp_method
        self.params = params
        with open(self.filepath) as _file:
            for i, smi in enumerate(_file):
                if i == 0:
                    # skip header of csv
                    continue
                self.smiles.append(smi.replace(",", ">").rstrip("\n"))

        # trivial labels for now
        self.labels = [0 for _ in self.smiles]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        descriptors = reaction_fps(self.smiles[idx],
                                   fp_method=self.fp_method,
                                   **self.params)

        return torch.from_numpy(descriptors).float().to(self.dev), self.labels[idx]


class ReactionSmartsTemplatesDataset(Dataset):
    alphabet = ['(', '[', '#', '1', '7', ']', '-', 'S', ';', 'H', '0', '+', ':', '*', '2', ')', '=', '3', '4',
                '.', '5', 'O', '6', '8', 'N', 'I', 'c', 'C', '9', 'l', 'n', 's', 'o', 'B', 'r', '@', 'F', '/',
                'P', 'a', '\\', 'b', 'i', 'Z', 'V', 'K', 'L', 'M', 'g', 'A', 'u', 'e', 'T', 'R', 'h', 'U']
    alphabet_dict = {char: i for i, char in enumerate(alphabet)}

    def __init__(self, filepath: str, dev: Any, binary: bool):
        self.filepath = filepath
        self.dev = dev
        self.binary = binary
        self.smarts_templates = []
        with open(self.filepath) as _file:
            for i, line in enumerate(_file):
                if i == 0:
                    # skip header of csv
                    continue
                self.smarts_templates.append(line.split(",")[1].rstrip())

        # trivial labels for now
        self.labels = [0 for _ in self.smarts_templates]

    def one_hot_encode(self, sm_temp):
        x = torch.zeros(len(self.alphabet))
        for char in sm_temp:
            if self.binary:
                x[self.alphabet_dict[char]] = 1.0
            else:
                x[self.alphabet_dict[char]] += 1.0
        return x

    def __len__(self):
        return len(self.smarts_templates)

    def __getitem__(self, idx):
        rxn = self.smarts_templates[idx]
        reags, prods = rxn.split(">>")
        reags_ohe = self.one_hot_encode(reags)
        prods_ohe = self.one_hot_encode(prods)
        if self.binary:
            res = torch.max(reags_ohe, prods_ohe)
        else:
            res = reags_ohe + prods_ohe
        return res.to(self.dev), self.labels[idx]
