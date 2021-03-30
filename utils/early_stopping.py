from typing import Union

from numpy import floor
from numpy.random import shuffle

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def split_train_val(train_dataset: Dataset, val_size: Union[float, int], batch_size: int):
    n_points = len(train_dataset)
    all_idx = list(range(n_points))
    shuffle(all_idx)
    if val_size < 1:
        split = int(floor(val_size * n_points))
    else:
        split = int(val_size)

    train_idx, val_idx = all_idx[split:], all_idx[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler)

    valid_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=validation_sampler,
                              num_workers=0)

    return train_loader, valid_loader
