import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset
from torch.optim.optimizer import Optimizer

from utils import EPS, get_q_joint


def loss_function(p_joint: Tensor, q_joint: Tensor) -> Tensor:
    """
    Calculates KLDiv between joint distributions in original and embedding space
    :param p_joint:
    :param q_joint:
    :return: KLDiv value
    """
    # TODO Add here alpha gradient calculation too
    return (p_joint * torch.log((p_joint + EPS) / (q_joint + EPS))).sum()


def fit_model(model: nn.Module,
              x_and_p_joint: TensorDataset,
              opt: Optimizer,
              n_epochs: int,
              batch_size: int) -> None:
    """
    Fit model, but the whole p_joint must be calculated
    beforehand and then fed in by batches
    :param model:
    :param x_and_p_joint:
    :param opt:
    :param n_epochs:
    :param batch_size:
    :return:
    """
    model.train()
    n_points = len(x_and_p_joint)
    for epoch in range(n_epochs):
        train_loss = 0
        for i in range(n_points // batch_size + 1):
            start_idx = i * batch_size
            fin_idx = start_idx + min(batch_size, n_points - start_idx)
            orig_points_batch, p_joint_batch = x_and_p_joint[start_idx: fin_idx]
            opt.zero_grad()
            embeddings = model(orig_points_batch)
            q_joint_batch = get_q_joint(embeddings, alpha=1, dist_func="euc")
            loss = loss_function(p_joint_batch, q_joint_batch)
            train_loss += loss.item()
            loss.backward()
            opt.step()
        print(f'====> Epoch: {epoch} Average loss: {train_loss / n_points:.4f}')


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal(m.weight.data)
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
