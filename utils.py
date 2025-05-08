import torch
import numpy as np

EPS = 1e-7

def to_list(e):
    if isinstance(e, torch.tensor) or isinstance(e, np.ndarray):
        return e.tolist()
    elif e is None:
        return None
    else:
        try:
            return list(e)
        except Exception:
            return [e]


def compute_gae(rewards: torch.tensor, values: torch.tensor, next_values: torch.tensor, gamma=0.99, lamb=0.9) -> torch.tensor:
    """Uses General Advantage Estimator to compute... general advantage estimation."""
    assert rewards.shape == values.shape == next_values.shape, f"r: {rewards.shape} - v: {values.shape} - v_: {next_values.shape}"
    delta = rewards + gamma * next_values - values
    gaes = torch.zeros_like(values)
    gaes[rewards.shape[0]-1] = delta[-1]
    for idx in reversed(range(rewards.shape[0])):
        gaes[idx-1] = delta[idx-1] + gamma * lamb * gaes[idx]
    return gaes


def normalize(t: torch.tensor, dim: int = 0) -> torch.tensor:
    """Returns normalized (zero 0 and std 1) tensor along specified axis (default: 0)."""
    if dim == 0:
        # Special case since by default it reduces on dim 0 and it should be faster.
        return (t - t.mean(dim=dim)) / torch.clamp(t.std(dim=dim), EPS)
    else:
        return (t - t.mean(dim=dim, keepdim=True)) / torch.clamp(t.std(dim=dim, keepdim=True), EPS)


def revert_norm_returns(rewards: torch.tensor, gamma: float = 0.99) -> torch.tensor:
    """
    Parameters:
        rewards: Rewards to discount. Expected shape (..., 1)
        dones: torch.tensor with termination flag. Expected ints {0, 1} in shape (..., 1)
        gamma: Discount factor.
    """
    discounted_reward = torch.zeros(rewards.shape[1:], dtype=rewards.dtype, device=rewards.device)
    returns = torch.zeros_like(rewards).float()
    len_returns = returns.shape[0]
    for idx, reward in enumerate(reversed(rewards)):
        discounted_reward = reward + gamma * discounted_reward
        returns[len_returns - idx - 1] = discounted_reward

    return normalize(returns, dim=0)