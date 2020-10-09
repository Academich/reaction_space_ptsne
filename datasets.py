import pickle
import gzip
from typing import Tuple, Optional
import numpy as np

import torch

# config
BATCH_SIZE = 1
NUM_WORKERS = 1
SHUFFLE = True


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