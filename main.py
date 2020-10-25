import torch
from datasets import SmilesDataset, load_mnist_some_classes, ReactionSmilesDataset
from torch.utils.data import TensorDataset
from model.model import NeuralMapping, fit_model, NeuralMappingDeeper
from plot_embeddings import plot_embs
from config import config

import datetime

if __name__ == '__main__':

    # Defining instruments
    if config.seed:
        torch.manual_seed(config.seed)
    dev = torch.device(config.dev)
    print(dev)

    # if config.data['name'] != "mnist":
    #     n_bits = config.data['n_bits']
    #     dataset_name = f"data/{config.data['name']}.smi"
    #
    #     points_ds = SmilesDataset(dataset_name, dev, n_bits=n_bits)
    #     dim_input = n_bits
    # else:
    #     include_classes = None
    #     points, labels = load_mnist_some_classes(include_classes)
    #     dim_input = points.size(1)
    #     points = points.to(dev)
    #     points_ds = TensorDataset(points, labels)
    # TODO это в конфиг
    path = "data/ibm-test.csv"
    fp_method = "structural"
    params = {"n_bits": 2048,
              "fp_type": "MorganFP",
              "include_agents": True,
              "agent_weight": 1,
              "non_agent_weight": 10,
              "bit_ratio_agents": 0.2
              }
    dim_input = 2048
    points_ds = ReactionSmilesDataset(path, dev, fp_method, params)

    # print(config.data['name'])
    net = NeuralMappingDeeper
    ffnn = net(dim_input=dim_input).to(dev)
    untrained_ref_ffnn = net(dim_input=dim_input).to(dev)
    opt = torch.optim.Adam(ffnn.parameters(), **config.optimization_conf)

    # Training and evaluating
    start = datetime.datetime.now()

    fit_model(ffnn, points_ds, opt, **config.training_params, save_model_flag=config.save_flag)

    fin = datetime.datetime.now()
    print("Training time:", fin - start)

    plot_embs(ffnn, untrained_ref_ffnn, points_ds)
