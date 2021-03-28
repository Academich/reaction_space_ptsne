from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer
)

from utils.reactions import reaction_fps

model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)


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
                if not params["include_agents"]:
                    reactants, agents, products = smi.split(">")
                    rearranged_smi = f"{reactants}.{agents}>>{products}"
                    self.smiles.append(rearranged_smi)
                else:
                    self.smiles.append(smi.strip())
                self.labels.append(int(label))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        descriptors = reaction_fps(self.smiles[idx],
                                   fp_method=self.fp_method,
                                   **self.params)

        return torch.from_numpy(descriptors).float().to(self.dev), self.labels[idx]


class BERTFpsReactionSmilesDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 no_agents: bool,
                 dev: Any) -> None:
        self.filepath = filepath  # path to .csv file
        self.dev = dev
        self.smiles = []
        self.labels = []
        self.fps_dict = {}
        with open(self.filepath) as _file:
            for i, line in enumerate(_file):
                try:
                    smi, label = line.split(";")
                except ValueError:
                    smi = line
                    label = 0
                smi = smi.strip()
                if no_agents:
                    reactants, agents, products = smi.split(">")
                    smi = f"{reactants}.{agents}>>{products}"
                self.smiles.append(smi)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if smi not in self.fps_dict:
            bert_fingerprint = rxnfp_generator.convert(self.smiles[idx])
            self.fps_dict[smi] = bert_fingerprint
        else:
            bert_fingerprint = self.fps_dict[smi]
        return torch.tensor(bert_fingerprint).float().to(self.dev), self.labels[idx]
