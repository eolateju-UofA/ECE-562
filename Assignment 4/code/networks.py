"""Actor and critic networks for SAC.

The actor is a tanh-squashed Gaussian:
    u ~ N(mu_theta(s), diag(sigma_theta(s)^2)),   a = tanh(u).

We call u the *pre-tanh* action and a the (squashed) action that is sent to the
environment. Hopper-v5 has action bounds [-1, 1], so no further rescaling is needed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def tanh_log_det_jacobian(u: torch.Tensor) -> torch.Tensor:
    """sum_i log(1 - tanh(u_i)^2), computed in a numerically stable way.

    Uses log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u)).
    Returns a tensor of shape (batch,).
    """
    return (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def dist_params(self, obs: torch.Tensor):
        """Mean and log-std of the pre-tanh Gaussian, each of shape (batch, act_dim)."""
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        """Reparameterized sample from the current policy.

        Returns a dict with
            a        squashed action tanh(u),                     (batch, act_dim)
            u        pre-tanh action,                             (batch, act_dim)
            logp_a   log pi_theta(a | s) (squashed density),      (batch,)
            logp_u   log N(u | mu, sigma) (pre-tanh density),      (batch,)
            mu, log_std                                           (batch, act_dim)
        Gradients flow through a, u, logp_a and logp_u (reparameterization trick).
        """
        mu, log_std = self.dist_params(obs)

        # ======================= TODO 1 =======================
        # (a) Build the pre-tanh Gaussian N(mu, diag(sigma^2)) with torch.distributions.Normal,
        #     sigma = exp(log_std), and draw a REPARAMETERIZED sample u (use rsample, not sample).
        # (b) Squash: a = tanh(u).
        # (c) logp_u = sum_i log N(u_i | mu_i, sigma_i)                       shape (batch,)
        #     logp_a = logp_u - sum_i log(1 - tanh(u_i)^2)                    shape (batch,)
        #     Use tanh_log_det_jacobian(u) for the second term (it is numerically stable).
        raise NotImplementedError("TODO 1: Actor.sample")
        # ======================================================

        return {"a": a, "u": u, "logp_a": logp_a, "logp_u": logp_u,
                "mu": mu, "log_std": log_std}

    def log_prob_u(self, obs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """log N(u | mu_theta(s), sigma_theta(s)) for a GIVEN pre-tanh action u. Shape (batch,).

        Useful for evaluating stored (replayed) actions under the current policy.
        """
        mu, log_std = self.dist_params(obs)
        return Normal(mu, log_std.exp()).log_prob(u).sum(dim=-1)

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.dist_params(obs)
        return torch.tanh(mu)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Q(s, a) for squashed actions a. Shape (batch,)."""
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)
