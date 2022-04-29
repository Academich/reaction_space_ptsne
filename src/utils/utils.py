from typing import Optional
from math import log2

import torch
from torch import Tensor

from src.utils.distance import distance_functions

EPS = 1e-10


def initialize_multiscale_p_joint(batch_size: int) -> 'Tensor':
    res = torch.zeros(batch_size, batch_size)
    return res


def calculate_optimized_p_cond(input_points: 'Tensor',
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

    n_points = input_points.size()[0]

    diag_mask = (1 - torch.eye(n_points)).type_as(input_points)

    dist_f = distance_functions[dist_func]
    distances = dist_f(input_points)

    # Binary search for optimal squared sigmas
    unit = torch.ones(n_points).type_as(input_points)
    min_sigma_sq = unit * (min_allowed_sig_sq + 1e-20)
    max_sigma_sq = unit * max_allowed_sig_sq
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
    if torch.isnan(ent_diff.max()):
        print("Warning! Entropy is nan. Discarding batch", flush=True)
        return
    return p_cond


def get_p_cond(distances: 'Tensor', sigmas_sq: 'Tensor', mask: 'Tensor') -> 'Tensor':
    """
    Calculates conditional probability distribution given distances and squared sigmas
    :param distances: Matrix of squared distances ||x_i - x_j||^2
    :param sigmas_sq: Row vector of squared sigma for each row in distances
    :param mask: A mask tensor to set diagonal elements to zero
    :return: Conditional probability matrix
    """
    eps = torch.tensor([EPS]).type_as(distances)
    logits = -distances / (2 * torch.max(sigmas_sq, eps).view(-1, 1))
    logits.exp_()
    masked_exp_logits = logits * mask
    normalization = torch.max(masked_exp_logits.sum(1), eps).unsqueeze(1)
    return masked_exp_logits / normalization + 1e-10


def get_q_joint(emb_points: 'Tensor', dist_func: str, alpha: int, ) -> 'Tensor':
    """
    Calculates the joint probability matrix in embedding space.
    :param emb_points: Points in embeddings space
    :param alpha: Number of degrees of freedom in t-distribution
    :param dist_func: A kay name for a distance function
    :return: Joint distribution matrix in emb. space
    """
    eps = torch.tensor([EPS]).type_as(emb_points)
    n_points = emb_points.size()[0]
    mask = (-torch.eye(n_points) + 1).to(emb_points.device)
    dist_f = distance_functions[dist_func]
    distances = dist_f(emb_points) / alpha
    q_joint = (1 + distances).pow(-(1 + alpha) / 2) * mask
    q_joint /= q_joint.sum()
    return torch.max(q_joint, eps)


def entropy(p: 'Tensor') -> 'Tensor':
    """
    Calculates Shannon Entropy for every row of a conditional probability matrix
    :param p: Conditional probability matrix, where every row sums up to 1
    :return: 1D tensor of entropies, (n_points,)
    """
    return -(p * p.log2()).sum(dim=1)


def make_joint(distr_cond: 'Tensor') -> 'Tensor':
    """
    Makes a joint probability distribution out of conditional distribution
    :param distr_cond: Conditional distribution matrix
    :return: Joint distribution matrix. All values in it sum up to 1.
    Too small values are set to fixed epsilon
    """
    eps = torch.tensor([EPS]).type_as(distr_cond)
    n_points = distr_cond.size()[0]
    distr_joint = (distr_cond + distr_cond.t()) / (2 * n_points)
    return torch.max(distr_joint, eps)


def calc_p_joint_in_batch(batch: 'Tensor',
                          dist_func_name: str,
                          perplexity: Optional[int],
                          bin_search_tol: float,
                          bin_search_max_iter: int,
                          min_allowed_sig_sq: int,
                          max_allowed_sig_sq: int):
    if perplexity is not None:
        target_entropy = log2(perplexity)
        p_cond_in_batch = calculate_optimized_p_cond(batch,
                                                     target_entropy,
                                                     dist_func_name,
                                                     bin_search_tol,
                                                     bin_search_max_iter,
                                                     min_allowed_sig_sq,
                                                     max_allowed_sig_sq)
        if p_cond_in_batch is None:
            return
        p_joint_in_batch = make_joint(p_cond_in_batch)

    else:
        _bs = batch.size()[0]
        max_entropy = round(log2(_bs / 2))
        n_different_entropies = 0
        mscl_p_joint_in_batch = torch.zeros(_bs, _bs).type_as(batch)
        for h in range(1, max_entropy):
            p_cond_for_h = calculate_optimized_p_cond(batch,
                                                      h,
                                                      dist_func_name,
                                                      bin_search_tol,
                                                      bin_search_max_iter,
                                                      min_allowed_sig_sq,
                                                      max_allowed_sig_sq)
            if p_cond_for_h is None:
                continue
            n_different_entropies += 1

            p_joint_for_h = make_joint(p_cond_for_h)
            mscl_p_joint_in_batch += p_joint_for_h

        p_joint_in_batch = mscl_p_joint_in_batch / n_different_entropies

    return p_joint_in_batch
