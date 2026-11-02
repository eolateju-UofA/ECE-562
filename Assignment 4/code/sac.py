"""Soft Actor-Critic (Haarnoja et al., 2018, with automatic temperature tuning).

SACAgent.update(batch) performs, in this order:
  1. a critic update on both Q-networks          (uses critic_loss,  TODO 2)
  2. an actor update                             (uses actor_loss,   TODO 3)
  3. a temperature (alpha) update                (uses alpha_loss,   TODO 4a)
  4. a Polyak update of the target critics       (soft_update,       TODO 4b)
The optimizer steps in update() are provided. You implement the four methods.
"""

import copy

import torch
import torch.nn.functional as F

from networks import Actor, QNetwork


class SACAgent:
    def __init__(self, obs_dim, act_dim, hidden=256, gamma=0.99, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, init_alpha=1.0,
                 device="cpu"):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(obs_dim, act_dim, hidden).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden).to(device)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)

        # Automatic entropy tuning with target entropy -|A|.
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(float(init_alpha), device=device).log().requires_grad_(True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature, detached from the graph."""
        return self.log_alpha.exp().detach()

    # ------------------------------------------------------------------
    def critic_loss(self, batch: dict):
        """Twin-critic soft Bellman loss.

        Returns (loss, q1_pred) where loss is a scalar tensor and q1_pred, of shape (batch,),
        is Q_phi1(s, a) on the batch (used for logging).
        """
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, terminated = batch["next_obs"], batch["terminated"]
        alpha = self.alpha

        # ======================= TODO 2 =======================
        # (a) Without gradients, sample a' ~ pi_theta(. | s') with ONE call to
        #     self.actor.sample(next_obs) and compute the soft Bellman target
        #       y = r + gamma * (1 - terminated) * [ min_j Q_target_j(s', a') - alpha * log pi(a' | s') ]
        # (b) Compute Q_phi1(s, a) and Q_phi2(s, a) for the STORED actions and return
        #       loss = 0.5 * ( MSE(Q_phi1, y) + MSE(Q_phi2, y) ).
        raise NotImplementedError("TODO 2: critic_loss")
        # ======================================================

    def actor_loss(self, obs: torch.Tensor):
        """Reparameterized SAC actor loss.

        Returns (loss, logp_a) where loss is a scalar tensor and logp_a, of shape (batch,), is
        log pi_theta(a~ | s) of the fresh actions (used by the temperature update).
        """
        alpha = self.alpha

        # ======================= TODO 3 =======================
        # Sample fresh actions a~ ~ pi_theta(. | s) with ONE call to self.actor.sample(obs)
        # (reparameterized, so gradients flow through a~) and return
        #       loss = mean[ alpha * log pi_theta(a~ | s) - min_j Q_phi_j(s, a~) ].
        # Use the online critics (not the targets). Do not update the critics here.
        raise NotImplementedError("TODO 3: actor_loss")
        # ======================================================

    def alpha_loss(self, logp_a: torch.Tensor) -> torch.Tensor:
        """Temperature loss. logp_a has shape (batch,) and must be treated as a constant."""
        # ======================= TODO 4a ======================
        #       loss = -mean[ log_alpha * ( log pi_theta(a~ | s) + target_entropy ) ]
        # Gradients must flow into self.log_alpha only.
        raise NotImplementedError("TODO 4a: alpha_loss")
        # ======================================================

    @torch.no_grad()
    def soft_update(self):
        """Polyak averaging of both target critics."""
        # ======================= TODO 4b ======================
        # For every parameter of q1 / q2 and its counterpart in q1_target / q2_target:
        #       phi_target <- (1 - tau) * phi_target + tau * phi
        raise NotImplementedError("TODO 4b: soft_update")
        # ======================================================

    # ------------------------------------------------------------------
    def update(self, batch: dict) -> dict:
        # 1. Critic
        critic_loss, q1_pred = self.critic_loss(batch)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. Actor (critic gradients created here are discarded: critic_opt.zero_grad()
        #    is called before every critic step)
        actor_loss, logp_a = self.actor_loss(batch["obs"])
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. Temperature
        alpha_loss = self.alpha_loss(logp_a)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4. Target networks
        self.soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": -logp_a.mean().item(),
            "q_mean": q1_pred.mean().item(),
        }
