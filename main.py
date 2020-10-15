import torch
from datasets import SmilesDataset
from model.model import NeuralMapping, fit_model
from plot_embeddings import plot_embs
from config import config

import datetime

if __name__ == '__main__':

    n_bits = 1024
    points_ds = SmilesDataset("data/nuclear.smi", n_bits=n_bits)
    dim_input = n_bits

    # Defining instruments
    if config.seed:
        torch.manual_seed(config.seed)
    dev = torch.device(config.dev)
    ffnn = NeuralMapping(dim_input=dim_input).to(dev)
    untrained_ref_ffnn = NeuralMapping(dim_input=dim_input).to(dev)
    opt = torch.optim.Adam(ffnn.parameters(), **config.optimization_conf)

    # Training and evaluating
    start = datetime.datetime.now()

    fit_model(ffnn, points_ds, opt, **config.training_params)
    plot_embs(ffnn, untrained_ref_ffnn, points_ds)

    fin = datetime.datetime.now()
    print("time elapsed:", fin - start)
