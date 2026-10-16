import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    """Diagonal-Gaussian actor with a state-independent log std, and a state-value critic."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = mlp(obs_dim, 1)
        self.actor_mean = mlp(obs_dim, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def actor_parameters(self):
        """Parameters of the policy: the mean network and the log standard deviation."""
        return list(self.actor_mean.parameters()) + [self.actor_logstd]

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """V(s), shape (batch_size,)."""
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """
        Returns (action, log pi(action | x), entropy, V(x)), each with a leading
        batch dimension. If `action` is None, an action is sampled.
        """
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        return action, log_prob, entropy, self.get_value(x)
