import torch
from torch import Tensor


def squared_euc_distances(x: 'Tensor') -> 'Tensor':
    """
    Calculates squared euclidean distances between rows
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared euclidean distances ||x_i - x_j||^2 (n_points, n_points)
    """
    sq_norms = (x ** 2).sum(dim=1)
    return sq_norms + sq_norms.unsqueeze(1) - 2 * x @ x.t()


def tanimoto_distances(x: 'Tensor') -> 'Tensor':
    sq_norms = (x ** 2).sum(dim=1)
    dot_product = x @ x.t()
    sum_of_distances = sq_norms + sq_norms.unsqueeze(1)
    similarity = dot_product / (sum_of_distances - dot_product)
    return 1 - similarity


def squared_tanimoto_distances(x: 'Tensor') -> 'Tensor':
    return tanimoto_distances(x) ** 2


def tanimoto_distances_binary(x: 'Tensor') -> 'Tensor':
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


def squared_tanimoto_distances_binary(x: 'Tensor') -> 'Tensor':
    """
    Calculates squared jaccard dissimilarities between rows.
    Rows should be binary vectors. Pretty fast function.
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    return tanimoto_distances_binary(x) ** 2


def cosine_distances(x: 'Tensor') -> 'Tensor':
    norms = torch.sqrt((x ** 2).sum(dim=1))
    dot_product = x @ x.t()
    similarity = dot_product / (norms * norms.unsqueeze(1))
    return 1 - similarity


def squared_cosine_distances(x: 'Tensor') -> 'Tensor':
    return cosine_distances(x) ** 2


distance_functions = {"euc": squared_euc_distances,
                      "tanimoto": squared_tanimoto_distances,
                      "tanimoto_binary": squared_tanimoto_distances_binary,
                      "cosine": squared_cosine_distances}
