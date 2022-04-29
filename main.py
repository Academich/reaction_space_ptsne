from argparse import ArgumentParser

import torch

import pytorch_lightning as pl

from src import RxnDataModule, PTSNEMapper

# TODO Instantiate loggers
# Wandb, RDKit

# === Initialize arguments parser ===
parser = ArgumentParser()

# PROGRAM level args
parser.add_argument("--seed", type=int, default=123456)
parser.add_argument("--save_dir_path", type=str, default="saved_models")

# add model specific args
parser = PTSNEMapper.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

# === Ensure reproducibility ===
pl.seed_everything(args.seed)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# === Instantiate classes
data_module = RxnDataModule.from_argparse_args(args)

dim_input = args.n_bits if args.fp_method != "transformer" else 256
model = PTSNEMapper.from_argparse_args(args, dim_input=dim_input)
trainer = pl.Trainer.from_argparse_args(args,
                                        logger=None,
                                        profiler=None,
                                        callbacks=None)

if __name__ == '__main__':
    trainer.fit(model, data_module)
