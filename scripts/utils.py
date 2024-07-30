import argparse
import torch

# Fucntion to convert strings to boolean variables (for reading cmd line args)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Function to calculate the log probability given a probabilistic network's (mean + log variance) output
def log_prob(targets, outputs, num_means=12, num_log_vars=12):
    # handle rnn batches
    if len(outputs.shape) > 2:
        # compress batch and trajectory dimensions
        outputs = outputs.view(-1, num_means+num_log_vars)
        targets = targets.view(-1, num_means)
        # remove zeroes (steps where no prediction was made)
        non_zero_mask = torch.any(targets != 0, dim=1)
        # Use the mask to filter out zero rows
        targets = targets[non_zero_mask]
        outputs = outputs[non_zero_mask]

    # parse outputs
    d = outputs.shape[-1]
    split_index = int(d/2)
    means = outputs[:,0:split_index]
    log_var = outputs[:,split_index:d]

    # calculate negative log prob of targets in outputs distributions
    diff = targets-means
    precision = torch.exp(-log_var)
    quadratic_term = -(0.5)*torch.sum(diff**2 * precision, dim=1)
    log_det_cov = torch.sum(log_var, dim=1)
    const_term = -0.5 * targets.shape[-1] * torch.log(torch.tensor([2 * torch.pi], device=log_var.device))

    log_probs = const_term - 0.5 * log_det_cov + quadratic_term

    # # torch sanity check
    # variances = torch.exp(log_var)
    # cov_matrices = torch.diag_embed(variances)
    # multivariate_normal_dists = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=cov_matrices)
    # torch_log_probs = multivariate_normal_dists.log_prob(targets)

    # sum and negate for optimization
    negative_log_probs_sum = -torch.sum(log_probs)

    return negative_log_probs_sum