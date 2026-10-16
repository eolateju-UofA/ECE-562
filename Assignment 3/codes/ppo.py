import torch
import torch.nn as nn


class PPOAgent:
    def __init__(self, actor_critic, clip_coef: float = 0.2, vf_coef: float = 0.5,
                 ent_coef: float = 0.0, max_grad_norm: float = 0.5, loss_type: str = "clip"):
        assert loss_type in ("clip", "ratio", "naive")
        self.ac = actor_critic
        self.loss_type = loss_type
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def update(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, optimizer):
        """One mini-batch gradient step (surrogate selected by loss_type + value loss)."""
        _, newlogprob, entropy, newvalue = self.ac.get_action_and_value(b_obs, b_actions)

        # Advantages were normalized once over the full rollout in compute_gae.
        # Do not renormalize them separately in each mini-batch.
        adv = b_advantages

        ratio = torch.exp(newlogprob - b_logprobs)
        if self.loss_type == "naive":
            # No correction: on-policy policy-gradient loss applied to stale data
            pg_loss = -(newlogprob * adv).mean()
        elif self.loss_type == "ratio":
            # Importance ratio only (unclipped surrogate)
            pg_loss = -(ratio * adv).mean()
        else:
            # Clipped surrogate objective
            pg_loss = -torch.min(ratio * adv,
                                 torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * adv).mean()

        # Value loss and total loss
        v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
        loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
        optimizer.step()
        return pg_loss.item(), v_loss.item(), entropy.mean().item()
