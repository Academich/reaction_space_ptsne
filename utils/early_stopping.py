from typing import Union

import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def split_train_val(train_dataset: Dataset, val_size: Union[float, int], batch_size: int, seed: int):
    n_points = len(train_dataset)
    all_idx = list(range(n_points))
    np.random.seed(seed)
    np.random.shuffle(all_idx)
    if val_size < 1:
        split = int(np.floor(val_size * n_points))
    else:
        split = int(val_size)

    n_folds = n_points // split
    for i in range(n_folds):
        print(f"Fold {i + 1}", flush=True)
        val_start_idx = split * i
        val_end_idx = split * (i + 1)
        val_end_idx = min(val_end_idx, n_points)
        val_idx = all_idx[val_start_idx:val_end_idx]
        train_idx = all_idx[:val_start_idx] + all_idx[val_end_idx:]

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler)

        valid_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler=validation_sampler,
                                  num_workers=0)

        yield train_loader, valid_loader
