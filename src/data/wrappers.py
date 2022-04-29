from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from src.utils import reaction_fps


class RxnDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 fp_method: str,
                 params: Dict[str, Any]) -> None:
        self.filepath = filepath  # path to .csv file
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

        return torch.from_numpy(descriptors).float(), self.labels[idx]


class RxnDataModule(pl.LightningDataModule):

    def __init__(self,
                 train_path: Optional[str],
                 val_path: Optional[str],
                 test_path: Optional[str],
                 batch_size: int,
                 num_workers: int,

                 fp_method: str,
                 fp_type: str,
                 n_bits: int,
                 include_agents: bool,
                 agent_weight: float,
                 non_agent_weight: float,
                 bit_ratio_agents: float):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.fp_method = fp_method
        self.fp_params = {"fp_type": fp_type,
                          "n_bits": n_bits,
                          "include_agents": include_agents,
                          "agent_weight": agent_weight,
                          "non_agent_weight": non_agent_weight,
                          "bit_ratio_agents": bit_ratio_agents}

    def prepare_data(self) -> None:
        # Use this method to do things that might write to disk or that need
        # to be done only from a single process in distributed settings.
        # download, tokenize, etc
        # called from a single process (e.g. GPU 0). Do not use it to assign state (self.x = y).
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # stage is used to separate setup logic for trainer.{fit,validate,test,predict}
        # if setup is called with stage = None, we assume all stages have been set up.
        # setup is called from every process
        # There are also data operations you might want to perform on every GPU. Use setup to do things like:
        # count number of classes
        # build vocabulary
        # perform train/val/test splits
        # apply transforms (defined explicitly in your datamodule)
        # etc…

        if stage == "fit" or stage is None:
            self.train = RxnDataset(self.train_path, self.fp_method, self.fp_params)
            self.val = RxnDataset(self.val_path, self.fp_method, self.fp_params)

        if stage == "test" or stage is None:
            self.test = RxnDataset(self.test_path, self.fp_method, self.fp_params)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
