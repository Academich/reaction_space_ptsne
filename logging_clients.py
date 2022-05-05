from typing import Optional

import wandb

from pytorch_lightning.loggers import WandbLogger


def make_wandb_logger(project_name: str, run_name: Optional[str] = None):
    wandb.init(project=project_name)
    if run_name is not None:
        wandb.run.name = run_name
        wandb.run.save()
    wandb_logger = WandbLogger()
    return wandb_logger
