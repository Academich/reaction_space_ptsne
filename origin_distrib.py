import torch
from math import log2

from utils import get_p_cond, entropy, make_joint, distance_functions


class OrigDistrBuilder:

    def __init__(self, input_points: torch.Tensor, perplexity: int):
        self.input_points = input_points  # .cuda()
        self.target_entropy = log2(perplexity)

    def calculate_optimized_p_cond(self,
                                   dist_func: str,
                                   tol: float = 1e-4,
                                   max_iter: int = 20):
        n_points = self.input_points.size(0)
        diag_mask = (1 - torch.eye(n_points))  # .cuda()

        dist_f = distance_functions[dist_func]
        distances = dist_f(self.input_points)

        # Binary search for optimal squared sigmas
        min_sigma_sq = 1e-20 * torch.ones(n_points)  # .cuda()
        max_sigma_sq = 1e2 * torch.ones(n_points)  # .cuda()
        sq_sigmas = (min_sigma_sq + max_sigma_sq) / 2
        p_cond = get_p_cond(distances, sq_sigmas, diag_mask)
        ent_diff = entropy(p_cond) - self.target_entropy
        finished = ent_diff.abs() < tol

        curr_iter = 0
        while curr_iter < max_iter and not finished.all().item():
            pos_diff = (ent_diff > 0).float()  # .cuda()
            neg_diff = (ent_diff <= 0).float()  # .cuda()

            max_sigma_sq = pos_diff * sq_sigmas + neg_diff * max_sigma_sq
            min_sigma_sq = pos_diff * min_sigma_sq + neg_diff * sq_sigmas

            sq_sigmas = finished.logical_not() * (min_sigma_sq + max_sigma_sq) / 2 + finished * sq_sigmas
            p_cond = get_p_cond(distances, sq_sigmas, diag_mask)
            ent_diff = entropy(p_cond) - self.target_entropy
            finished = ent_diff.abs() < tol

            curr_iter += 1

        return p_cond


if __name__ == '__main__':
    a = torch.tensor([[0., 0., 0], [0., 1., 0], [0., 2., 0], [0., 3., 0]])
    print(a)
    p_cond_builder = OrigDistrBuilder(a, perplexity=2)
    p_cond = p_cond_builder.calculate_optimized_p_cond(dist_func="euc")
    print(p_cond)
    p_joint = make_joint(p_cond)
    print(p_joint)
