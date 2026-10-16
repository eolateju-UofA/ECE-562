import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # State-value function critic backbone
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Continuous policy mean backbone
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        
        # State-independent learnable log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        # =====================================================================
        # TODO 1: Continuous Distribution Utilities (Q3)
        # 1. Evaluate the joint log probability of the action:
        #    Sum across the action dimensions: log_prob.sum(dim=-1)
        # 2. Evaluate the joint differential entropy of the normal distribution:
        #    Sum across the action dimensions: probs.entropy().sum(dim=-1)
        # Ensure returned tensors have shape (batch_size,).
        # =====================================================================
        log_prob = probs.log_prob(action).sum(dim=-1)  # Student implementation
        entropy = probs.entropy().sum(dim=-1)   # Student implementation
        # =====================================================================
        
        return action, log_prob, entropy, self.critic(x)
