import torch

class RolloutBuffer:
    def __init__(self, num_steps: int, obs_dim: int, act_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.device = device
        
        # Rollout storage for sequential single-environment transitions
        self.obs = torch.zeros((num_steps, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, act_dim), device=device)
        self.logprobs = torch.zeros(num_steps, device=device)
        self.rewards = torch.zeros(num_steps, device=device)
        self.dones = torch.zeros(num_steps, device=device)
        self.values = torch.zeros(num_steps, device=device)
        
        self.advantages = torch.zeros(num_steps, device=device)
        self.returns = torch.zeros(num_steps, device=device)
        self.step = 0

    def insert(self, obs, action, logprob, reward, done, value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.squeeze()
        self.step += 1

    def reset(self):
        self.step = 0

    def compute_gae(self, next_value: torch.Tensor, next_done: float, gamma: float, gae_lambda: float):
        """
        TODO 2: Sequential Generalized Advantage Estimation (Q4)
        Iterate backwards from t = num_steps - 1 down to 0:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_{t+1}) - V(s_t)
            A_t = delta_t + gamma * lambda * (1 - d_{t+1}) * A_{t+1}
            R_t = A_t + V(s_t)
            
        Notice that at boundary step t = num_steps - 1:
            V(s_{t+1}) is next_value
            d_{t+1} is next_done
        """
        # =====================================================================
        last_gae_lam = 0.0
        for t in reversed(range(self.num_steps)):
            # Student implementation here:
            if t == self.num_steps - 1: # Boundary: use bootstrap values
                next_val = next_value.squeeze()
                next_d = next_done
            else: # Interior: use stored values/dones from buffer
                next_val = self.values[t + 1]
                next_d = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_val * (1 - next_d) - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * (1 - next_d) * last_gae_lam
            self.advantages[t] = last_gae_lam

            # pass

        self.returns = self.advantages + self.values
        # =====================================================================