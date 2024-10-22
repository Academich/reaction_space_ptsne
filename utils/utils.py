import random
import string
from math import log2
from torch import tensor, eye, ones, device, isnan, zeros
from torch import max as torch_max
from config import config

EPS = tensor([1e-10]).to(device(config.dev))


def get_random_string(length: int) -> str:
    """Generates random string of ascii chars"""
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str


def squared_euc_dists(x: tensor) -> tensor:
    """
    Calculates squared euclidean distances between rows
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared euclidean distances ||x_i - x_j||^2 (n_points, n_points)
    """
    sq_norms = (x ** 2).sum(dim=1)
    return sq_norms + sq_norms.unsqueeze(1) - 2 * x @ x.t()


def jaccard_distances(x: tensor) -> tensor:
    """
    Calculates jaccard dissimilarities between rows.
    Rows should be binary vectors. Pretty fast function.
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    n_ones = x.sum(dim=1)
    intersection = x @ x.t()
    sum_of_ones = n_ones + n_ones.unsqueeze(1)
    similarity = intersection / (sum_of_ones - intersection)
    return 1 - similarity


def squared_jaccard_distances(x: tensor) -> tensor:
    """
    Calculates squared jaccard dissimilarities between rows.
    Rows should be binary vectors. Pretty fast function.
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    return jaccard_distances(x) ** 2


def general_jaccard_distances(x: tensor) -> tensor:
    """
    Calculates jaccard dissimilarities between rows.
    Rows may be continuous vectors. Pretty slow function
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    lesser = (x < x.unsqueeze(1))
    greater = lesser.__invert__()
    lesser = lesser.float()
    greater = greater.float()
    minimums = (x * lesser + x.unsqueeze(1) * greater).sum(2)
    maximums = (x * greater + x.unsqueeze(1) * lesser).sum(2)
    similarity = minimums / maximums
    return 1 - similarity


def squared_general_jaccard_distances(x: tensor) -> tensor:
    """
    Calculates squared jaccard dissimilarities between rows.
    Rows may be continuous vectors. Pretty slow function
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    return general_jaccard_distances(x) ** 2


def squared_cosine_distances(x: tensor) -> tensor:
    raise NotImplementedError


distance_functions = {"euc": squared_euc_dists,
                      "jaccard": squared_jaccard_distances,
                      "jaccard_general": squared_general_jaccard_distances,
                      "cosine": squared_cosine_distances}


def initialize_multiscale_p_joint(batch_size: int) -> tensor:
    res = zeros(batch_size, batch_size).to(device(config.dev))
    return res


def calculate_optimized_p_cond(input_points: tensor,
                               target_entropy: float,
                               dist_func: str,
                               tol: float,
                               max_iter: int,
                               min_allowed_sig_sq: float,
                               max_allowed_sig_sq: float):
    """
    Adjust sigmas for every row in conditional probability matrix
    to match the given perplexity using binary search
    :param input_points: Unoptimized conditional probability matrix
    :param target_entropy: The entropy that every distribution (row) in conditional
    probability matrix will be optimized to match
    :param dist_func: A string denoting the desired distance function
    :param tol: The tolerance threshold for binary search
    :param max_iter: Number of maximum iterations for binary search
    :param min_allowed_sig_sq: Minimum allowed value for squared sigmas
    :param max_allowed_sig_sq: Maximum allowed value for squared sigmas
    :return: Conditional probability matrix optimized to match the given perplexity
    """

    n_points = input_points.size(0)

    diag_mask = (1 - eye(n_points)).to(device(config.dev))

    dist_f = distance_functions[dist_func]
    distances = dist_f(input_points)

    # Binary search for optimal squared sigmas
    min_sigma_sq = (min_allowed_sig_sq + 1e-20) * ones(n_points).to(device(config.dev))
    max_sigma_sq = max_allowed_sig_sq * ones(n_points).to(device(config.dev))
    sq_sigmas = (min_sigma_sq + max_sigma_sq) / 2
    p_cond = get_p_cond(distances, sq_sigmas, diag_mask)
    ent_diff = entropy(p_cond) - target_entropy
    finished = ent_diff.abs() < tol

    curr_iter = 0
    while not finished.all().item():
        if curr_iter >= max_iter:
            print(f"Warning! Exceeded max iter. Not optimized: {(~finished).sum().item()}", flush=True)
            # print("Discarding batch")
            return p_cond
        pos_diff = (ent_diff > 0).float()
        neg_diff = (ent_diff <= 0).float()

        max_sigma_sq = pos_diff * sq_sigmas + neg_diff * max_sigma_sq
        min_sigma_sq = pos_diff * min_sigma_sq + neg_diff * sq_sigmas

        sq_sigmas = finished.logical_not() * (min_sigma_sq + max_sigma_sq) / 2 + finished * sq_sigmas
        p_cond = get_p_cond(distances, sq_sigmas, diag_mask)
        ent_diff = entropy(p_cond) - target_entropy
        finished = ent_diff.abs() < tol
        curr_iter += 1
    if isnan(ent_diff.max()):
        print("Warning! Entropy is nan. Discarding batch", flush=True)
        return
    return p_cond


def get_p_cond(distances: tensor, sigmas_sq: tensor, mask: tensor) -> tensor:
    """
    Calculates conditional probability distribution given distances and squared sigmas
    :param distances: Matrix of squared distances ||x_i - x_j||^2
    :param sigmas_sq: Row vector of squared sigma for each row in distances
    :param mask: A mask tensor to set diagonal elements to zero
    :return: Conditional probability matrix
    """
    logits = -distances / (2 * torch_max(sigmas_sq, EPS).view(-1, 1))
    logits.exp_()
    masked_exp_logits = logits * mask
    normalization = torch_max(masked_exp_logits.sum(1), EPS).unsqueeze(1)
    return masked_exp_logits / normalization + 1e-10


def get_q_joint(emb_points: tensor, dist_func: str, alpha: int, ) -> tensor:
    """
    Calculates the joint probability matrix in embedding space.
    :param emb_points: Points in embeddings space
    :param alpha: Number of degrees of freedom in t-distribution
    :param dist_func: A kay name for a distance function
    :return: Joint distribution matrix in emb. space
    """
    n_points = emb_points.size(0)
    mask = (-eye(n_points) + 1).to(emb_points.device)
    dist_f = distance_functions[dist_func]
    distances = dist_f(emb_points) / alpha
    q_joint = (1 + distances).pow(-(1 + alpha) / 2) * mask
    q_joint /= q_joint.sum()
    return torch_max(q_joint, EPS)


def entropy(p: tensor) -> tensor:
    """
    Calculates Shannon Entropy for every row of a conditional probability matrix
    :param p: Conditional probability matrix, where every row sums up to 1
    :return: 1D tensor of entropies, (n_points,)
    """
    return -(p * p.log2()).sum(dim=1)


def make_joint(distr_cond: tensor) -> tensor:
    """
    Makes a joint probability distribution out of conditional distribution
    :param distr_cond: Conditional distribution matrix
    :return: Joint distribution matrix. All values in it sum up to 1.
    Too small values are set to fixed epsilon
    """
    n_points = distr_cond.size(0)
    distr_joint = (distr_cond + distr_cond.t()) / (2 * n_points)
    return torch_max(distr_joint, EPS)
