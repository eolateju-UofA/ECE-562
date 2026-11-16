"""Soft Actor-Critic (Haarnoja et al., 2018, with automatic temperature tuning).

SACAgent.update(batch) performs, in this order:
  1. a critic update on both Q-networks          (uses critic_loss,  TODO 2)
  2. an actor update                             (uses actor_loss,   TODO 3)
  3. a temperature (alpha) update                (uses alpha_loss,   TODO 4a)
  4. a Polyak update of the target critics       (soft_update,       TODO 4b)
The optimizer steps in update() are provided. You implement the four methods.
"""

import copy
import math

import torch
import torch.nn.functional as F

from networks import Actor, QNetwork

PPO_CLIP_EPS = 0.2  # epsilon used by the clip diagnostics (Q4) and the clip actor loss (Q5b)


class SACAgent:
    def __init__(self, obs_dim, act_dim, hidden=256, gamma=0.99, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, init_alpha=1.0,
                 device="cpu", target_action="current", actor_loss_type="reparam"):
        # Section 2 options (defaults reproduce standard SAC):
        #   target_action   "current" (a' ~ pi_theta, TODO 2) | "replay" (a' = stored next action)
        #   actor_loss_type "reparam" (TODO 3) | "naive" | "clip"
        assert target_action in ("current", "replay")
        assert actor_loss_type in ("reparam", "naive", "clip")
        self.target_action = target_action
        self.actor_loss_type = actor_loss_type
        self._actor_stats = {}
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
        with torch.no_grad():
            nxt = self.actor.sample(next_obs)
            a_next, logp_next = nxt["a"], nxt["logp_a"]
            if self.target_action == "replay":
                # Q5(a): use the action the behavior policy actually took at s' and its
                # squashed log-density under pi_b. Fall back to the fresh a' ~ pi_theta when
                # there is no stored next action (time-limit cut, or not collected yet).
                has = batch["has_next"].bool()
                a_next = torch.where(has[:, None], batch["next_act"], a_next)
                logp_next = torch.where(has, batch["next_beh_logp_a"], logp_next)
            q_next = torch.min(self.q1_target(next_obs, a_next),
                               self.q2_target(next_obs, a_next))
            y = rew + self.gamma * (1.0 - terminated) * (q_next - alpha * logp_next)
        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        loss = 0.5 * F.mse_loss(q1_pred, y) + 0.5 * F.mse_loss(q2_pred, y)
        return loss, q1_pred
        # ======================================================

    def actor_loss(self, obs: torch.Tensor, batch: dict = None):
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
        out = self.actor.sample(obs)
        if self.actor_loss_type == "reparam":
            q = torch.min(self.q1(obs, out["a"]), self.q2(obs, out["a"]))
            loss = (alpha * out["logp_a"] - q).mean()
            return loss, out["logp_a"]
        return self._ppo_style_actor_loss(obs, batch, out), out["logp_a"]
        # ======================================================

    def _ppo_style_actor_loss(self, obs, batch, out):
        """Q5(b): naive / clip policy-gradient losses on the STORED actions.

        A_hat(s, a) = Qbar(s, a) - Qbar(s, a~), computed without gradients and normalized over
        the batch as in Assignment 3. The entropy term alpha * E[log pi_theta(a~ | s)] uses the
        fresh reparameterized sample, exactly as in SAC.
        """
        alpha = self.alpha
        act, u = batch["act"], batch["pre_tanh"]
        with torch.no_grad():
            q_data = torch.min(self.q1(obs, act), self.q2(obs, act))
            q_fresh = torch.min(self.q1(obs, out["a"]), self.q2(obs, out["a"]))
            adv = q_data - q_fresh
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # log pi_theta(a | s) and log N_theta(u | s) differ by a term that does not depend on
        # theta, so the pre-tanh density gives the same gradient.
        logp_u_new = self.actor.log_prob_u(obs, u)
        entropy_term = alpha * out["logp_a"].mean()

        if self.actor_loss_type == "naive":
            pg_loss = -(logp_u_new * adv).mean()
        else:  # clip
            log_ratio = logp_u_new - batch["beh_logp_u"]
            ratio = log_ratio.exp()
            eps = PPO_CLIP_EPS
            pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
            with torch.no_grad():
                outside = (log_ratio > math.log(1 + eps)) | (log_ratio < math.log(1 - eps))
                self._actor_stats["clipfrac_actor"] = outside.float().mean().item()
        return pg_loss + entropy_term

    # ------------------------------------------------------------------
    @torch.no_grad()
    def diagnostics(self, batch: dict) -> dict:
        """Q4(a): how off-policy is the sampled batch? Computed before any parameter update.

        Only reads the networks (no sampling, no gradients), so it cannot change training.
        Skipped for batches without behavior information (e.g. the unit tests).
        """
        if "cur_step" not in batch or "step" not in batch:
            return {}
        stats = {"data_age_mean": (batch["cur_step"] - batch["step"]).float().mean().item()}

        keep = batch["is_random"] < 0.5          # leave out uniform warm-up samples
        if not keep.any():
            return stats
        obs = batch["obs"][keep]
        mu, log_std = self.actor.dist_params(obs)
        mu_b, log_std_b = batch["beh_mu"][keep], batch["beh_log_std"][keep]

        # KL( N_b || N_theta ) for diagonal Gaussians, summed over action dimensions.
        kl = (log_std - log_std_b
              + (torch.exp(2 * log_std_b) + (mu_b - mu) ** 2) / (2 * torch.exp(2 * log_std))
              - 0.5).sum(dim=-1)
        stats["kl_beh"] = kl.mean().item()

        # log r = log N_theta(u | s) - log N_b(u | s); the tanh terms cancel.
        logp_u_new = torch.distributions.Normal(mu, log_std.exp()).log_prob(
            batch["pre_tanh"][keep]).sum(dim=-1)
        log_ratio = logp_u_new - batch["beh_logp_u"][keep]
        q = torch.quantile(log_ratio, torch.tensor([0.05, 0.5, 0.95], device=log_ratio.device))
        stats["log_ratio_p05"], stats["log_ratio_p50"], stats["log_ratio_p95"] = q.tolist()

        # Fraction outside [1 - eps, 1 + eps], tested in log space to avoid overflow.
        eps = PPO_CLIP_EPS
        outside = (log_ratio > math.log(1 + eps)) | (log_ratio < math.log(1 - eps))
        stats["clipfrac_beh"] = outside.float().mean().item()
        return stats

    def alpha_loss(self, logp_a: torch.Tensor) -> torch.Tensor:
        """Temperature loss. logp_a has shape (batch,) and must be treated as a constant."""
        # ======================= TODO 4a ======================
        #       loss = -mean[ log_alpha * ( log pi_theta(a~ | s) + target_entropy ) ]
        # Gradients must flow into self.log_alpha only.
        return -(self.log_alpha * (logp_a.detach() + self.target_entropy)).mean()
        # ======================================================

    @torch.no_grad()
    def soft_update(self):
        """Polyak averaging of both target critics."""
        # ======================= TODO 4b ======================
        # For every parameter of q1 / q2 and its counterpart in q1_target / q2_target:
        #       phi_target <- (1 - tau) * phi_target + tau * phi
        for net, targ in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            for p, p_targ in zip(net.parameters(), targ.parameters()):
                p_targ.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
        # ======================================================

    # ------------------------------------------------------------------
    def update(self, batch: dict) -> dict:
        # 0. Diagnostics on the batch, before any parameter changes (Q4a)
        diag = self.diagnostics(batch)

        # 1. Critic
        critic_loss, q1_pred = self.critic_loss(batch)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. 
        #  (critic gradients created here are discarded: critic_opt.zero_grad()
        #    is called before every critic step)
        self._actor_stats = {}
        actor_loss, logp_a = self.actor_loss(batch["obs"], batch)
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
            **diag,
            **self._actor_stats,
        }
