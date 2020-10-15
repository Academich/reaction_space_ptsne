import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from utils import EPS, get_q_joint, calculate_optimized_p_cond, make_joint


def loss_function(p_joint: Tensor, q_joint: Tensor) -> Tensor:
    """
    Calculates KLDiv between joint distributions in original and embedding space
    :param p_joint:
    :param q_joint:
    :return: KLDiv value
    """
    # TODO Add here alpha gradient calculation too
    # TODO Add L2-penalty for early compression?
    return (p_joint * torch.log((p_joint + EPS) / (q_joint + EPS))).sum()


def fit_model(model: nn.Module,
              input_points: Dataset,
              opt: Optimizer,
              perplexity: int,
              n_epochs: int,
              early_exaggeration: int,
              early_exaggeration_constant: int,
              batch_size: int,
              dist_func_name: str,
              bin_search_tol: float,
              bin_search_max_iter: int,
              min_allowed_sig_sq: float,
              max_allowed_sig_sq: float
              ) -> None:
    """
    Fits t-SNE model
    :param model: nn.Module instance
    :param input_points: tensor of original points
    :param opt: optimizer instance
    :param perplexity: perplexity
    :param n_epochs: Number of epochs for training
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
    :return:
    """
    model.train()
    n_points = len(input_points)
    train_dl = DataLoader(input_points, batch_size=batch_size, shuffle=True)
    for epoch in range(n_epochs):
        train_loss = 0
        for list_with_batch in train_dl:
            orig_points_batch, _ = list_with_batch  # because it is a list of len 1
            with torch.no_grad():
                p_cond_in_batch = calculate_optimized_p_cond(orig_points_batch,
                                                             perplexity,
                                                             dist_func_name,
                                                             bin_search_tol,
                                                             bin_search_max_iter,
                                                             min_allowed_sig_sq,
                                                             max_allowed_sig_sq)
                p_joint_in_batch = make_joint(p_cond_in_batch)
            opt.zero_grad()
            embeddings = model(orig_points_batch)
            q_joint_in_batch = get_q_joint(embeddings, dist_func_name, alpha=1)
            if early_exaggeration:
                p_joint_in_batch *= early_exaggeration_constant
                early_exaggeration -= 1
            loss = loss_function(p_joint_in_batch, q_joint_in_batch)
            train_loss += loss.item()
            loss.backward()
            opt.step()
        print(f'====> Epoch: {epoch + 1} Average loss: {train_loss / n_points:.4f}')


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


class NeuralMapping(nn.Module):

    def __init__(self, dim_input, dim_emb=2):
        super().__init__()
        self.linear_1 = nn.Linear(dim_input, dim_input)
        self.bn_1 = nn.BatchNorm1d(dim_input)
        self.linear_2 = nn.Linear(dim_input, dim_emb)
        self.relu = nn.ReLU()

        self.apply(weights_init)

    def forward(self, x):
        x = self.relu(self.bn_1(self.linear_1(x)))
        return self.linear_2(x)
