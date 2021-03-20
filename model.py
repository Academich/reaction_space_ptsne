import datetime
import json
import os
from tqdm import tqdm

import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from numpy import array
from numpy import save as np_save

from utils.utils import EPS, get_q_joint, calculate_optimized_p_cond, make_joint, get_random_string


def loss_function(p_joint: Tensor, q_joint: Tensor) -> Tensor:
    """
    Calculates KLDiv between joint distributions in original and embedding space
    :param p_joint:
    :param q_joint:
    :return: KLDiv value
    """
    return (p_joint * torch.log((p_joint + EPS) / (q_joint + EPS))).sum()


def fit_model(model: torch.nn.Module,
              input_points: Dataset,
              opt: Optimizer,
              perplexity: int,
              n_epochs: int,
              save_dir_path: str,
              epochs_to_save_after: int,
              early_exaggeration: int,
              early_exaggeration_constant: int,
              batch_size: int,
              dist_func_name: str,
              bin_search_tol: float,
              bin_search_max_iter: int,
              min_allowed_sig_sq: float,
              max_allowed_sig_sq: float,
              configuration_report: str
              ) -> None:
    """
    Fits t-SNE model
    :param model: nn.Module instance
    :param input_points: tensor of original points
    :param opt: optimizer instance
    :param perplexity: perplexity
    :param n_epochs: Number of epochs for training
    :param save_dir_path: path to directory to save a trained model to
    :param epochs_to_save_after: number of epochs to save a model after. If passed None,
    model won't be saved at all
    :param early_exaggeration: Number of first training cycles in which
    exaggeration will be applied
    :param early_exaggeration_constant: Constant by which p_joint is multiplied in early exaggeration
    :param batch_size: Batch size for training
    :param dist_func_name: Name of distance function for distance matrix.
    Possible names: "euc", "jaccard", "cosine"
    :param bin_search_tol: Tolerance threshold for binary search to obtain p_cond
    :param bin_search_max_iter: Number of max iterations for binary search
    :param min_allowed_sig_sq: Minimal allowed value for squared sigmas
    :param max_allowed_sig_sq: Maximal allowed value for squared sigmas
    :param configuration_report: Config of the model in string form for report purposes
    :return:
    """
    model.train()
    batches_passed = 0
    train_dl = DataLoader(input_points, batch_size=batch_size, shuffle=True)
    model_name = get_random_string(6)
    epoch_losses = []
    for epoch in range(n_epochs):
        epoch_start_time = datetime.datetime.now()
        train_loss = 0
        for list_with_batch in tqdm(train_dl):
            orig_points_batch, _ = list_with_batch
            with torch.no_grad():
                p_cond_in_batch = calculate_optimized_p_cond(orig_points_batch,
                                                             perplexity,
                                                             dist_func_name,
                                                             bin_search_tol,
                                                             bin_search_max_iter,
                                                             min_allowed_sig_sq,
                                                             max_allowed_sig_sq)
                if p_cond_in_batch is None:
                    continue
                p_joint_in_batch = make_joint(p_cond_in_batch)
            opt.zero_grad()
            batches_passed += 1
            embeddings = model(orig_points_batch)
            q_joint_in_batch = get_q_joint(embeddings, "euc", alpha=1)
            if early_exaggeration:
                p_joint_in_batch *= early_exaggeration_constant
                early_exaggeration -= 1
            loss = loss_function(p_joint_in_batch, q_joint_in_batch)
            train_loss += loss.item()
            loss.backward()
            opt.step()
        epoch_end_time = datetime.datetime.now()
        time_elapsed = epoch_end_time - epoch_start_time

        # Report loss for epoch
        average_loss = train_loss / batches_passed
        epoch_losses.append(average_loss)
        print(f'====> Epoch: {epoch + 1}. Time {time_elapsed}. Average loss: {average_loss:.4f}', flush=True)

        # Save model and loss history if needed
        save_path = os.path.join(save_dir_path, f"{model_name}_epoch_{epoch + 1}")
        if epochs_to_save_after is not None and (epoch + 1) % epochs_to_save_after == 0:
            torch.save(model, save_path + ".pt")
            with open(save_path + ".json", "w") as here:
                json.dump(json.loads(configuration_report), here)
            print('Model saved as %s' % save_path, flush=True)

        if epochs_to_save_after is not None and epoch == n_epochs - 1:
            epoch_losses = array(epoch_losses)
            loss_save_path = save_path + "_loss.npy"
            np_save(loss_save_path, epoch_losses)
            print("Loss history saved in", loss_save_path, flush=True)


def get_batch_embeddings(model: nn.Module,
                         input_points: Dataset,
                         batch_size: int,
                         ) -> Tensor:
    """
    Yields final embeddings for every batch in dataset
    :param model:
    :param input_points:
    :param batch_size:
    :return:
    """
    model.eval()
    test_dl = DataLoader(input_points, batch_size=batch_size, shuffle=False)
    for batch_points, batch_labels in test_dl:
        with torch.no_grad():
            embeddings = model(batch_points)
            yield embeddings, batch_labels


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.BatchNorm1d,)):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


class NeuralMapper(nn.Module):

    def __init__(self, dim_input, dim_emb=2):
        super().__init__()
        self.linear_1 = nn.Linear(dim_input, dim_input)
        self.bn_1 = nn.BatchNorm1d(dim_input)
        self.linear_2 = nn.Linear(dim_input, dim_input)
        self.bn_2 = nn.BatchNorm1d(dim_input)
        self.linear_3 = nn.Linear(dim_input, dim_input)
        self.bn_3 = nn.BatchNorm1d(dim_input)
        self.linear_4 = nn.Linear(dim_input, dim_emb)
        self.relu = nn.ReLU()

        self.apply(weights_init)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.linear_2(self.relu(x))
        x = self.bn_2(x)
        x = self.linear_3(self.relu(x))
        x = self.bn_3(x)
        x = self.linear_4(self.relu(x))
        return x
