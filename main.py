import torch
from torch.utils.data import TensorDataset

from datasets import ReactionSmilesDataset, SmilesDataset, load_mnist_some_classes
from model.model import fit_model, NeuralMappingDeeper
from visual_evaluation.plot_embeddings import plot_embs
from config import config

import datetime

if __name__ == '__main__':

    # Defining instruments
    if config.seed:
        torch.manual_seed(config.seed)
    dev = torch.device(config.dev)
    print(dev)
    print(config.problem)

    if config.problem == "molecules":
        settings = config.problem_settings["molecules"]
        n_bits = settings['n_bits']
        dataset_name = f"data/{settings['filename']}"
        points_ds = SmilesDataset(dataset_name, dev, n_bits=n_bits)
        dim_input = n_bits
        print(dataset_name)
    elif config.problem == "reactions":
        settings = config.problem_settings["reactions"]
        path = f"data/{settings['filename']}"
        print(path)
        fp_method = settings["fp_method"]
        params = {"n_bits": settings["n_bits"],
                  "fp_type": settings["fp_type"],
                  "include_agents": settings["include_agents"],
                  "agent_weight": settings["agent_weight"],
                  "non_agent_weight": settings["non_agent_weight"],
                  "bit_ratio_agents": settings["bit_ratio_agents"]
                  }
        dim_input = settings["n_bits"]
        points_ds = ReactionSmilesDataset(path, dev, fp_method, params)
    elif config.problem == "mnist":
        include_classes = None
        points, labels = load_mnist_some_classes(include_classes)
        dim_input = points.size(1)
        points = points.to(dev)
        points_ds = TensorDataset(points, labels)
    else:
        raise ValueError(f"Unknown problem type: {config.problem}")

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
