import wandb

from pytorch_lightning.loggers import WandbLogger


def make_wandb_logger(project_name: str):
    wandb.init(project=project_name)
    wandb_logger = WandbLogger()
    return wandb_logger
