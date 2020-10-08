from torch import Tensor, tensor, eye
from torch import max as torch_max

EPS = tensor([1e-10])  # .cuda()


def squared_euc_dists(x: Tensor) -> Tensor:
    """
    Calculates squared euclidean distances between rows
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared euclidean distances ||x_i - x_j||^2 (n_points, n_points)
    """
    sq_norms = (x ** 2).sum(dim=1)
    return sq_norms + sq_norms.unsqueeze(1) - 2 * x @ x.t()


def squared_jaccard_distances(x: Tensor) -> Tensor:
    raise NotImplementedError


def squared_cosine_distances(x: Tensor) -> Tensor:
    raise NotImplementedError


distance_functions = {"euc": squared_euc_dists,
                      "jaccard": squared_jaccard_distances,
                      "cosine": squared_cosine_distances}


def get_p_cond(distances: Tensor, sigmas_sq: Tensor, mask: Tensor) -> Tensor:
    """
    Calculates conditional probability distribution given distances and squared sigmas
    :param distances: Matrix of squared distances ||x_i - x_j||^2
    :param sigmas_sq: Row vector of squared sigma for each row in distances
    :param mask: A mask tensor to set diagonal elements to zero
    :return: Conditional probability matrix
    """
    logits = -distances / (2 * sigmas_sq.view(-1, 1))
    logits.exp_()
    masked_exp_logits = logits * mask
    normalization = masked_exp_logits.sum(1).unsqueeze(1)
    return masked_exp_logits / normalization + 1e-10


def get_q_joint(emb_points: Tensor, alpha: int, dist_func: str) -> Tensor:
    """
    Calculates the joint probability matrix in embedding space.
    :param emb_points: Points in embeddings space
    :param alpha: Number of degrees of freedom in t-distribution
    :param dist_func: A kay name for a distance function
    :return: Joint distribution matrix in emb. space
    """
    n_points = emb_points.size(0)
    mask = (-eye(n_points) + 1)  # .cuda()
    dist_f = distance_functions[dist_func]
    distances = dist_f(emb_points)
    q_joint = (1 + distances).pow(-(1 + alpha) / 2) * mask
    q_joint /= q_joint.sum()
    return torch_max(q_joint, EPS)


def entropy(p: Tensor) -> Tensor:
    """
    Calculates Shannon Entropy for every row of a conditional probability matrix
    :param p: Conditional probability matrix, where every row sums up to 1
    :return: 1D tensor of entropies, (n_points,)
    """
    return -(p * p.log2()).sum(dim=1)


def make_joint(distr_cond: Tensor) -> Tensor:
    """
    Makes a joint probability distribution out of conditional distribution
    :param distr_cond: Conditional distribution matrix
    :return: Joint distribution matrix. All values in it sum up to 1.
    Too small values are set to fixed epsilon
    """
    n_points = distr_cond.size(0)
    distr_joint = (distr_cond + distr_cond.t()) / (2 * n_points)
    return torch_max(distr_joint, EPS)
