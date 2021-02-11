import json
import datetime
import argparse

import torch

from datasets import ReactionSmilesDataset
from model import fit_model, NeuralMapper
from config import config

parser = argparse.ArgumentParser()
parser.add_argument('--perplexity', '-p', type=int, default=None,
                    help='perplexity to use instead of one in the config')
parser.add_argument('--epochs', '-e', type=int, default=None,
                    help='perplexity to use instead of one in the config')
parser.add_argument('--batchsize', '-b', type=int, default=None,
                    help='batch size to use instead of one in the config')

args = parser.parse_args()

if __name__ == '__main__':

    # Defining instruments
    if config.seed:
        torch.manual_seed(config.seed)
    dev = torch.device(config.dev)
    print(dev, flush=True)
    print(config.problem, flush=True)

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

    net = NeuralMapper
    ffnn = net(dim_input=dim_input).to(dev)
    untrained_ref_ffnn = net(dim_input=dim_input).to(dev)
    opt = torch.optim.Adam(ffnn.parameters(), **config.optimization_conf)

    # Training and evaluating
    start = datetime.datetime.now()

    if args.perplexity is not None:
        config.training_params["perplexity"] = args.perplexity
    if args.epochs is not None:
        config.training_params["n_epochs"] = args.epochs
    if args.batchsize is not None:
        config.training_params["batch_size"] = args.batchsize

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
