import torch
import torch.nn as nn
from actor_critic import ActorCritic

class PPOAgent:
    def __init__(
        self,
        actor_critic: ActorCritic,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5
    ):
        self.ac = actor_critic
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def update(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, optimizer):
        """
        Executes one mini-batch gradient step for PPO.
        """
        _, newlogprob, entropy, newvalue = self.ac.get_action_and_value(b_obs, b_actions)
        newvalue = newvalue.view(-1)
        
        # =====================================================================
        # TODO 3: Clipped Surrogate Policy & Value Loss (Q5)
        # 1. Standardize b_advantages over the mini-batch:
        #    b_advantages = (b_advantages - mean) / (std + 1e-8)
        # 2. Compute probability ratio:
        #    r_t(theta) = exp(newlogprob - b_logprobs)
        # 3. Compute clipped surrogate policy loss:
        #    pg_loss = -E[ min(r_t * A, clip(r_t, 1-eps, 1+eps) * A) ]
        # 4. Compute MSE value function loss:
        #    v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
        # 5. Compute total loss:
        #    loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy.mean()
        # =====================================================================

        # 1. Advantage normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 2. Probability ratio
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        # 3. Clipped surrogate policy loss
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)

        pg_loss = torch.max(pg_loss1, pg_loss2).mean()   # Student implementation
        v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()  # Student implementation
        loss = pg_loss + (self.vf_coef * v_loss) - (self.ent_coef * entropy.mean())      # Student implementation
        # =====================================================================

        optimizer.zero_grad()
        loss.backward()

        # =====================================================================
        # TODO 4: Gradient Clipping (Q5)
        # Clip network gradients using torch.nn.utils.clip_grad_norm_
        # with maximum norm self.max_grad_norm
        # =====================================================================
        # Student implementation here:
        nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
        # =====================================================================
        
        optimizer.step()
        return loss.item(), pg_loss.item(), v_loss.item(), entropy.mean().item()
