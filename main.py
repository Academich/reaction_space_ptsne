import json
import torch
from torch.utils.data import TensorDataset

from datasets import ReactionSmilesDataset, SmilesDataset, load_mnist_some_classes, ReactionSmartsTemplatesDataset
from model.model import fit_model, NeuralMapper
from visual_evaluation.plot_embeddings import plot_embs
from config import config

import datetime

if __name__ == '__main__':

    # Defining instruments
    if config.seed:
        torch.manual_seed(config.seed)
    dev = torch.device(config.dev)
    print(dev, flush=True)
    print(config.problem, flush=True)

    if config.problem == "molecules":
        settings = config.problem_settings["molecules"]
        n_bits = settings['n_bits']
        dataset_name = f"data/{settings['filename']}"
        points_ds = SmilesDataset(dataset_name, dev, n_bits=n_bits)
        dim_input = n_bits
        print(dataset_name, flush=True)
    elif config.problem == "reactions":
        settings = config.problem_settings["reactions"]
        path = f"data/{settings['filename']}"
        print(path, flush=True)
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
    elif config.problem == "reaction_templates":
        settings = config.problem_settings["reaction_templates"]
        path = f"data/{settings['filename']}"
        print(path, flush=True)
        points_ds = ReactionSmartsTemplatesDataset(path, dev, binary=settings["binary"])
        dim_input = len(points_ds.alphabet)
        print(dim_input, flush=True)

    elif config.problem == "mnist":
        settings = None
        include_classes = None
        points, labels = load_mnist_some_classes(include_classes)
        dim_input = points.size(1)
        points = points.to(dev)
        points_ds = TensorDataset(points, labels)
    else:
        raise ValueError(f"Unknown problem type: {config.problem}")

    net = NeuralMapper
    ffnn = net(dim_input=dim_input).to(dev)
    untrained_ref_ffnn = net(dim_input=dim_input).to(dev)
    opt = torch.optim.Adam(ffnn.parameters(), **config.optimization_conf)

    # Training and evaluating
    start = datetime.datetime.now()

    report_config = json.dumps({"settings": settings,
                                "optimization": config.optimization_conf,
                                "training": config.training_params})
    fit_model(ffnn,
              points_ds,
              opt,
              **config.training_params,
              save_model_flag=config.save_flag,
              configuration_report=report_config)

    fin = datetime.datetime.now()
    print("Training time:", fin - start, flush=True)

    # plot_embs(ffnn, untrained_ref_ffnn, points_ds)
