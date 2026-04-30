import torch 
import torch.nn.functional as F 
from tensordict import TensorDict 

#TODO: I don't know what this is. 
def soft_ce(pred, target, cfg):
    """
    Computes the cross entropy loss between predictions and soft targets.
    """

    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)

def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1.0)

def gaussian_logprob(eps, log_std):
    """
    Compute Gaussian log probability.
    """
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385332046727 # log(sqrt(2 * pi))
    return log_prob.sum(-1, keepdim=True)

def squash(mu, pi, log_pi):
    """Apply squashing function. The idea of this squashing function is to make sure that the policy can not take
    actions that are greater than the bounds of the action space.
    adapted from https://github.com/nicklashansen/tdmpc2
    """
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1-pi.pow(2)) + 1e-6) # add small epsilon to prevent log(0)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi

def int_to_one_hot(x, num_classes):
    """
    Convers an integer tensor to a one-hot tensor.
    Supports batched inputs.
    """
    one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    one_hot.scatter_(-1, x.unsqueeze(-1), 1.0)
    return one_hot


def symlog(x):
    """
    Symmetric logarithm function. 
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def two_hot(x, cfg):
    """
    converts a batch of scalars to soft two-hot encoded targets for discrete regression.
    """
    if cfg.num_bins == 0:
        return x 
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1-bin_offset)
    soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot

def two_hot_inv(x, cfg):
    """
    Converts a batch of soft two-hot encoded vectors to scalars.
    """
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x*dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
    """
    Sample from the Gumbel-Softmax distribution
    """

    logits = p.log()
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim=dim)
    return y_soft.argmax(-1)

    