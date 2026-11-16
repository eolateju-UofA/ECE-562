"""FIFO replay buffer for SAC.

Besides the usual (s, a, r, s', terminated) tuple, every transition also stores
information about the BEHAVIOR policy that collected it (analogous to how the
Assignment 3 rollout buffer stored log pi_theta_old):

    pre_tanh     u = atanh(a), the pre-tanh action                     (act_dim,)
    beh_mu       mean of the behavior's pre-tanh Gaussian at s         (act_dim,)
    beh_log_std  log-std of the behavior's pre-tanh Gaussian at s      (act_dim,)
    beh_logp_u   log-density of u under the behavior policy            scalar
                 (pre-tanh density, i.e. WITHOUT the tanh Jacobian)
    step         environment step at which the transition was collected
    is_random    1 if the action came from the uniform warm-up policy

Added for Q5(a) (the "replay" critic target), filled in by set_next() once the agent
picks the action at s' in the same episode:
    next_act         a'_b, the action the behavior policy took at s'           (act_dim,)
    next_beh_logp_a  log pi_b(a'_b | s'), squashed density (WITH the tanh term) scalar
    has_next         1 if next_act is available, 0 otherwise (episode ended, or the
                     next step has not been collected yet)

For warm-up transitions the behavior is Uniform([-1, 1]^m) in action space, so
beh_mu / beh_log_std are meaningless placeholders (zeros) and beh_logp_u is the
exact pre-tanh density of the uniform behavior:
    log p(u) = -m log 2 + sum_i log(1 - tanh(u_i)^2).

The default capacity (10^6) exceeds the number of environment steps in this
assignment, so by default nothing is ever evicted.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0      # next write position
        self.size = 0     # number of valid transitions

        f32 = np.float32
        self.obs = np.zeros((self.capacity, obs_dim), dtype=f32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=f32)
        self.act = np.zeros((self.capacity, act_dim), dtype=f32)
        self.pre_tanh = np.zeros((self.capacity, act_dim), dtype=f32)
        self.rew = np.zeros((self.capacity,), dtype=f32)
        self.terminated = np.zeros((self.capacity,), dtype=f32)
        self.beh_mu = np.zeros((self.capacity, act_dim), dtype=f32)
        self.beh_log_std = np.zeros((self.capacity, act_dim), dtype=f32)
        self.beh_logp_u = np.zeros((self.capacity,), dtype=f32)
        self.step = np.zeros((self.capacity,), dtype=np.int64)
        self.is_random = np.zeros((self.capacity,), dtype=f32)
        self.next_act = np.zeros((self.capacity, act_dim), dtype=f32)
        self.next_beh_logp_a = np.zeros((self.capacity,), dtype=f32)
        self.has_next = np.zeros((self.capacity,), dtype=f32)

    def add(self, obs, act, pre_tanh, rew, next_obs, terminated,
            beh_mu, beh_log_std, beh_logp_u, step, is_random):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.pre_tanh[i] = pre_tanh
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)
        self.beh_mu[i] = beh_mu
        self.beh_log_std[i] = beh_log_std
        self.beh_logp_u[i] = beh_logp_u
        self.step[i] = step
        self.is_random[i] = float(is_random)
        self.has_next[i] = 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def set_next(self, i: int, next_act, next_beh_logp_a: float):
        """Record the behavior policy's action at s' for the transition stored at position i."""
        self.next_act[i] = next_act
        self.next_beh_logp_a[i] = next_beh_logp_a
        self.has_next[i] = 1.0

    def sample_idx(self, idx: np.ndarray) -> dict:
        """Gather the transitions at buffer positions idx into a dict of tensors."""
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return {
            "idx": idx,
            "obs": t(self.obs), "act": t(self.act), "pre_tanh": t(self.pre_tanh),
            "rew": t(self.rew), "next_obs": t(self.next_obs),
            "terminated": t(self.terminated),
            "beh_mu": t(self.beh_mu), "beh_log_std": t(self.beh_log_std),
            "beh_logp_u": t(self.beh_logp_u),
            "step": t(self.step), "is_random": t(self.is_random),
            "next_act": t(self.next_act), "next_beh_logp_a": t(self.next_beh_logp_a),
            "has_next": t(self.has_next),
        }

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        """Uniformly sample batch_size transitions (with replacement)."""
        idx = rng.integers(0, self.size, size=batch_size)
        return self.sample_idx(idx)
