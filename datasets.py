from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from utils.reactions import reaction_fps


class ReactionSmilesDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 dev: Any,
                 fp_method: str,
                 params: Dict[str, Any]) -> None:
        self.filepath = filepath  # path to .csv file
        self.dev = dev
        self.smiles = []
        self.labels = []
        self.fp_method = fp_method
        self.params = params
        with open(self.filepath) as _file:
            for i, line in enumerate(_file):
                try:
                    smi, label = line.split(";")
                except ValueError:
                    smi = line
                    label = 0
                self.smiles.append(smi.strip())
                self.labels.append(int(label))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        descriptors = reaction_fps(self.smiles[idx],
                                   fp_method=self.fp_method,
                                   **self.params)

        return torch.from_numpy(descriptors).float().to(self.dev), self.labels[idx]
