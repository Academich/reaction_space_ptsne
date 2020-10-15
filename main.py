import torch
from torch.utils.data import TensorDataset
from datasets import load_mnist_some_classes
from model.model import NeuralMapping, fit_model, get_batch_embeddings
from plot_embeddings import plot_embs
from config import config

import datetime

if __name__ == '__main__':
    # Defining data
    include_classes = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    points, labels = load_mnist_some_classes(include_classes)
    dim_input = points.size(1)

    # Defining instruments
    if config.seed:
        torch.manual_seed(config.seed)
    dev = torch.device(config.dev)
    ffnn = NeuralMapping(dim_input=dim_input).to(dev)
    untrained_ref_ffnn = NeuralMapping(dim_input=dim_input).to(dev)
    opt = torch.optim.Adam(ffnn.parameters(), **config.optimization_conf)
    points = points.to(dev)
    points_ds = TensorDataset(points)

    # Training and evaluating
    start = datetime.datetime.now()

    fit_model(ffnn, points_ds, opt, **config.training_params)
    plot_embs(ffnn, untrained_ref_ffnn, points_ds)

    fin = datetime.datetime.now()
    print("time elapsed:", fin - start)
