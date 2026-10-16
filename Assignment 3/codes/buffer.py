import torch


class RolloutBuffer:
    """
    Sequential single-environment rollout storage.

    Convention: dones[t] = 1 if obs[t] is the first observation of a new episode
    (i.e. the previous transition ended the episode). With this convention,
    dones[t + 1] is the termination flag d_{t+1} used in the GAE recursion.
    """

    def __init__(self, num_steps: int, obs_dim: int, act_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.device = device

        self.obs = torch.zeros((num_steps, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, act_dim), device=device)
        self.logprobs = torch.zeros(num_steps, device=device)   # log pi(a_t | s_t) of the policy that collected the data
        self.rewards = torch.zeros(num_steps, device=device)
        self.dones = torch.zeros(num_steps, device=device)
        self.values = torch.zeros(num_steps, device=device)     # V(s_t) at collection time

        self.advantages = torch.zeros(num_steps, device=device)
        self.returns = torch.zeros(num_steps, device=device)

        # Bootstrap information for the state s_T after the last transition
        self.next_obs = torch.zeros(obs_dim, device=device)
        self.next_done = 0.0
        self.next_value = torch.zeros((), device=device)        # V(s_T) at collection time
        self.step = 0

    def insert(self, obs, action, logprob, reward, done, value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.squeeze()
        self.step += 1

    def finish(self, next_obs: torch.Tensor, next_done: float, next_value: torch.Tensor):
        self.next_obs = next_obs.clone()
        self.next_done = float(next_done)
        self.next_value = next_value.reshape(())

    @torch.no_grad()
    def refresh_values(self, actor_critic):
        """
        Re-evaluate V(s_t) for all stored states and V(s_T) for the bootstrap state
        with the current critic. Observations, actions, rewards, dones and the
        behavior log-probabilities are left untouched.
        """
        self.values[:] = actor_critic.get_value(self.obs)
        self.next_value = actor_critic.get_value(self.next_obs.unsqueeze(0)).reshape(())

    def compute_gae(self, gamma: float, gae_lambda: float):
        """
        Backward GAE recursion using the values stored in self.values and self.next_value:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_{t+1}) - V(s_t)
            A_t     = delta_t + gamma * lambda * (1 - d_{t+1}) * A_{t+1}
            R_t     = A_t + V(s_t)
        """
        last_gae_lam = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - self.next_done
                nextvalues = self.next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * nextnonterminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns[:] = self.advantages + self.values

        # Normalize advantages once over the full rollout.  Keep returns based
        # on the unnormalized GAE values so the critic target is unchanged.
        self.advantages[:] = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
