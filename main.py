import torch.nn.functional as F
from torch import optim, tensor, matmul, sum, ones, cat, zeros
from torch.utils.data import TensorDataset

from datasets import load_mnist_some_classes
from origin_distrib import OrigDistrBuilder
from utils import make_joint

import datetime

if __name__ == '__main__':
    include_classes = (1, 2, 8)
    points, labels = load_mnist_some_classes(include_classes)
    start = datetime.datetime.now()
    p_cond_builder = OrigDistrBuilder(points, perplexity=30)
    p_cond = p_cond_builder.calculate_optimized_p_cond(dist_func="euc")
    p_joint = make_joint(p_cond)
    fin = datetime.datetime.now()
    print("time elapsed:", fin - start)
