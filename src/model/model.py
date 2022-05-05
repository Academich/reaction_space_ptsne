from typing import Optional
import inspect
from argparse import ArgumentParser

import torch
from torch import nn
from torch import Tensor

import pytorch_lightning as pl

from src.utils import calc_p_joint_in_batch, get_q_joint


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.BatchNorm1d,)):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


def kl_div(p_joint: 'Tensor', q_joint: 'Tensor') -> 'Tensor':
    """
    Calculates KLDiv between joint distributions in original and embedding space
    :param p_joint:
    :param q_joint:
    :return: KLDiv value
    """
    eps = 1e-10
    return (p_joint * torch.log((p_joint + eps) / (q_joint + eps))).sum()


class PTSNEMapper(pl.LightningModule):

    def __init__(self,
                 learning_rate: float,
                 dim_input: int,
                 dist_func_name: str,
                 perplexity: Optional[int],
                 bin_search_tol: float,
                 bin_search_max_iter: int,
                 min_allowed_sig_sq: int,
                 max_allowed_sig_sq: int,
                 early_exaggeration: Optional[int]):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        d_in = self.hparams.dim_input
        d_out = 2  # t-sne projection to a plane
        self.model = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.BatchNorm1d(d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in),
            nn.BatchNorm1d(d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in),
            nn.BatchNorm1d(d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_out),
        )
        self.model.apply(weights_init)

        self.criterion = kl_div

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        orig_points_batch, _ = batch
        projection = self.forward(orig_points_batch)
        # With torch no grad?
        p_joint_in_batch = calc_p_joint_in_batch(orig_points_batch,
                                                 self.hparams.dist_func_name,
                                                 self.hparams.perplexity,
                                                 self.hparams.bin_search_tol,
                                                 self.hparams.bin_search_max_iter,
                                                 self.hparams.min_allowed_sig_sq,
                                                 self.hparams.max_allowed_sig_sq)
        q_joint_in_batch = get_q_joint(projection, "euc", alpha=1)
        if self.hparams.early_exaggeration is not None:
            p_joint_in_batch *= self.hparams.early_exaggeration
            self.hparams.early_exaggeration -= 1

        loss = self.criterion(p_joint_in_batch, q_joint_in_batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        orig_points_batch, _ = batch
        projection = self.forward(orig_points_batch)
        p_joint_in_batch = calc_p_joint_in_batch(orig_points_batch,
                                                 self.hparams.dist_func_name,
                                                 self.hparams.perplexity,
                                                 self.hparams.bin_search_tol,
                                                 self.hparams.bin_search_max_iter,
                                                 self.hparams.min_allowed_sig_sq,
                                                 self.hparams.max_allowed_sig_sq)
        q_joint_in_batch = get_q_joint(projection, "euc", alpha=1)
        if self.hparams.early_exaggeration is not None:
            p_joint_in_batch *= self.hparams.early_exaggeration
            self.hparams.early_exaggeration -= 1

        loss = self.criterion(p_joint_in_batch, q_joint_in_batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    # === Argparse-related methods ===
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PTSNEMapper")
        # Model
        parser.add_argument("--learning_rate", type=float, default=0.001,
                            help="Constant learning rate for the Adam optimizer.")
        parser.add_argument("--dist_func_name", type=str, default="euc",
                            help="Distance function to use in t-SNE. "
                                 "Options: euc, tanimoto, tanimoto_binary, cosine.")
        parser.add_argument("--perplexity", type=int, default=None,
                            help="Perplexity parameter for t-SNE. If not specified, multi-scale t-SNE is run.")
        parser.add_argument("--bin_search_tol", type=float, default=0.0001,
                            help="Tolerance threshold for binary search.")
        parser.add_argument("--bin_search_max_iter", type=int, default=100,
                            help="Maximum allowed number of iterations of binary search.")
        parser.add_argument("--min_allowed_sig_sq", type=int, default=0,
                            help="Minimum allowed dispersion of the gaussian bells used in the domain space.")
        parser.add_argument("--max_allowed_sig_sq", type=int, default=10000,
                            help="Maximum allowed dispersion of the gaussian bells used in the domain space.")
        parser.add_argument("--early_exaggeration", type=int, default=None,
                            help="Early exaggeration constant. "
                                 "Increases attractive forces between points in the start of the training. "
                                 "Not used by default.")
        # DataModule
        parser.add_argument("--train_path", type=str, default=None,
                            help="Training data path.")
        parser.add_argument("--val_path", type=str, default=None,
                            help="Validation data path.")
        parser.add_argument("--test_path", type=str, default=None,
                            help="Test data path.")
        parser.add_argument("--batch_size", type=int, default=1024,
                            help="Batch size.")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="Number of workers for the data loader.")
        # DataModule fingerprints-related
        parser.add_argument("--fp_method", type=str, default="difference",
                            help="Type of fingerprints to represent reactions. "
                                 "Options: difference, structural, transformer. "
                                 "If 'transformer is chosen', all the following arguments are not used.")
        parser.add_argument("--fp_type", type=str, default="MorganFP",
                            help="Type of RDKit fingerprints of individual molecules. "
                                 "Not used with fp_method=transformer. "
                                 "Options: MorganFP, TopologicalTorsion, AtomPairFP - "
                                 "these three work for both difference and structural fingerprints. "
                                 "Some other RDKit fingerprints work only for fp_method=structural.")
        parser.add_argument("--n_bits", type=int, default=2048,
                            help="Number of bits in molecular fingerprints.")
        parser.add_argument("--include_agents", action='store_true', default=False,
                            help="Whether to include fingerprints of reagents into the reaction fingerprint.")
        parser.add_argument("--agent_weight", type=float, default=1,
                            help="Weight of the contribution of the fingerprints of the reagent "
                                 "molecules to the reaction difference fingerprint.")
        parser.add_argument("--non_agent_weight", type=float, default=1,
                            help="Weight of the contribution of the fingerprints of the reactant and product "
                                 "molecules to the reaction difference fingerprint."
                            )
        parser.add_argument("--bit_ratio_agents", type=float, default=0.2,
                            help="Relative number of reagent molecules bits in the reaction structural fingerprint.")
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)

        params = vars(args)

        # We only want to pass in valid class args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        trainer_kwargs = {name: params[name] for name in valid_kwargs if name in params}
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)