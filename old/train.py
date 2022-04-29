import json
import datetime
import argparse
import torch

from datasets import ReactionSmilesDataset, BERTFpsReactionSmilesDataset
from model import fit_model, NeuralMapper
from utils.early_stopping import split_train_val
from config import Config, get_conf_dict

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='config.yaml', 
                    help='path to configuration file')
parser.add_argument('--perplexity', '-p', type=int, default=None,
                    help='perplexity to use instead of one in the config')
parser.add_argument('--epochs', '-e', type=int, default=None,
                    help='perplexity to use instead of one in the config')
parser.add_argument('--batchsize', '-b', type=int, default=None,
                    help='batch size to use instead of one in the config')

args = parser.parse_args()

cfg_path = get_conf_dict(args.config)
config = Config(cfg_path)

if __name__ == '__main__':

    # Defining instruments
    if config.seed:
        torch.manual_seed(config.seed)
    dev = torch.device(config.dev)
    print(dev, flush=True)

    settings = config.problem_settings["reactions"]
    train_path = f"data/{settings['train_filename']}"
    val_path = f"data/{settings['val_filename']}" if settings['val_filename'] is not None else None
    val_points_ds = None

    print("Train:", train_path, flush=True)
    if val_path is not None:
        print("Validation:", val_path, flush=True)
    fp_method = settings["fp_method"]
    if fp_method == "transformer":
        dim_input = 256
        no_agents = settings["no_agents"]
        points_ds = BERTFpsReactionSmilesDataset(train_path, no_agents, dev)
        if val_path is not None:
            val_points_ds = BERTFpsReactionSmilesDataset(val_path, no_agents, dev)
    else:
        params = {"n_bits": settings["n_bits"],
                  "fp_type": settings["fp_type"],
                  "include_agents": settings["include_agents"],
                  "agent_weight": settings["agent_weight"],
                  "non_agent_weight": settings["non_agent_weight"],
                  "bit_ratio_agents": settings["bit_ratio_agents"]
                  }
        dim_input = settings["n_bits"]
        points_ds = ReactionSmilesDataset(train_path, dev, fp_method, params)
        if val_path is not None:
            val_points_ds = ReactionSmilesDataset(val_path, dev, fp_method, params)

    net = NeuralMapper
    ffnn = net(dim_input=dim_input).to(dev)
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

    bsize = config.training_params["batch_size"]
    if val_path is None:
        train_dl, val_dl = split_train_val(points_ds,
                                           val_size=150000,
                                           batch_size=bsize,
                                           seed=config.seed)
    else:
        train_dl, _ = split_train_val(points_ds,
                                      val_size=0,
                                      batch_size=bsize,
                                      seed=config.seed)
        val_dl, _ = split_train_val(val_points_ds,
                                    val_size=0,
                                    batch_size=bsize,
                                    seed=config.seed)
    fit_model(ffnn,
              train_dl,
              val_dl,
              opt,
              **config.training_params,
              epochs_to_save_after=config.epochs_to_save_after,
              save_dir_path=config.save_dir_path,
              configuration_report=report_config)

    fin = datetime.datetime.now()
    print("Training time:", fin - start, flush=True)
