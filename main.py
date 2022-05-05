from argparse import ArgumentParser

import torch

import pytorch_lightning as pl

from src import RxnDataModule, PTSNEMapper

# === Initialize arguments parser ===
parser = ArgumentParser()

# PROGRAM level args
parser.add_argument("--seed", type=int, default=123456,
                    help="Random seed for reproducibility.")
parser.add_argument("--save_dir_path", type=str, default="saved_models",
                    help="Path to the folder with the saved models.")
parser.add_argument("--use_wandb", action="store_true", default=False,
                    help="A flag whether to use W&B for logging or not.")
parser.add_argument("--wandb_project_name", type=str, default="reaction-space-ptsne",
                    help="The project name for W&B.")
parser.add_argument("--wandb_run_name", type=str, default=None,
                    help="The name of the run. Used by W&B and as a model checkpoint name.")

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

# === Callbacks ===
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.save_dir_path,
                                                   verbose=True,
                                                   save_last=True)
progress_bar_callback = pl.callbacks.progress.TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)
cb_list = [checkpoint_callback, progress_bar_callback]

# === Instantiate classes
data_module = RxnDataModule.from_argparse_args(args)

dim_input = args.n_bits if args.fp_method != "transformer" else 256
model = PTSNEMapper.from_argparse_args(args, dim_input=dim_input)

# === Logger clients (W&B or Tensorboard)
logger, run_name = None, args.wandb_run_name
if args.use_wandb:
    import logging_clients
    logger = logging_clients.make_wandb_logger(args.wandb_project_name, run_name)

trainer = pl.Trainer.from_argparse_args(args,
                                        logger=logger,
                                        profiler=None,
                                        callbacks=cb_list)

if __name__ == '__main__':
    trainer.fit(model, data_module)
