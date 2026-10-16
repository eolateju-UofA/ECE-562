"""
Basic PPO for Hopper-v5 (ECE 562, Assignment 3).

    python train.py --seed 1
    python train.py --seed 1 --staleness 2 --loss ratio

Writes one CSV row per policy update to <log-dir>/<run-name>.csv.
"""
import argparse
import collections
import csv
import os

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from actor_critic import ActorCritic
from buffer import RolloutBuffer
from ppo import PPOAgent

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Observation/reward preprocessing
# ---------------------------------------------------------------------------
class Normalizer:
    """
    Identity preprocessing for Assignment 3.

    Running observation normalization and reward scaling are intentionally
    disabled so that the stale-data experiments isolate policy staleness
    rather than changes in normalization statistics.  The interface is kept
    unchanged to minimize changes to the starter code.
    """

    def __init__(self, obs_dim, gamma, clip=10.0):
        self.gamma = gamma

    def obs(self, o, update=False):
        return np.asarray(o, dtype=np.float32)

    def reward(self, r, running_return, update=False):
        return float(r)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------
class EnvRunner:
    """Steps one environment with a policy and keeps the environment state across rollouts."""

    def __init__(self, env, seed, normalizer, update_stats):
        self.env = env
        self.norm = normalizer
        self.update_stats = update_stats
        self.disc_return = 0.0
        obs, _ = env.reset(seed=seed)
        self.obs = self._to_tensor(obs)
        self.done = 0.0                 # 1.0 if self.obs is the first observation of an episode
        self.ep_return = 0.0
        self.finished_returns = []      # undiscounted returns of completed episodes

    @torch.no_grad()
    def collect(self, ac, num_steps):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        buf = RolloutBuffer(num_steps, obs_dim, act_dim, torch.device("cpu"))
        for _ in range(num_steps):
            action, logprob, _, value = ac.get_action_and_value(self.obs.unsqueeze(0))
            next_obs, reward, terminated, truncated, _ = self.env.step(action.squeeze(0).numpy())
            self.disc_return = self.disc_return * self.norm.gamma + reward
            scaled_reward = self.norm.reward(reward, self.disc_return, self.update_stats)
            buf.insert(self.obs, action.squeeze(0), logprob.squeeze(0), scaled_reward, self.done, value)
            self.ep_return += reward
            done_flag = float(terminated or truncated)
            if done_flag:
                self.finished_returns.append(self.ep_return)
                self.ep_return, self.disc_return = 0.0, 0.0
                next_obs, _ = self.env.reset()
            self.obs = self._to_tensor(next_obs)
            self.done = done_flag
        buf.finish(self.obs, self.done, ac.get_value(self.obs.unsqueeze(0)))
        return buf

    def _to_tensor(self, obs):
        return torch.tensor(self.norm.obs(obs, self.update_stats), dtype=torch.float32)


@torch.no_grad()
def evaluate_policy(ac, eval_env, num_episodes, normalizer):
    """Average undiscounted return of the deterministic policy mean."""
    returns = []
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done, total = False, 0.0
        while not done:
            obs_t = torch.tensor(normalizer.obs(obs), dtype=torch.float32).unsqueeze(0)
            action = ac.actor_mean(obs_t).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return returns


@torch.no_grad()
def ratio_stats(ac, rollout, clip_coef, suffix):
    """
    Compares the current policy with the policy that collected `rollout`
    (whose log-probabilities are stored in rollout.logprobs):
    approximate KL(behavior || current), ratio percentiles and clip fraction.
    """
    _, newlogprob, _, _ = ac.get_action_and_value(rollout.obs, rollout.actions)
    logratio = newlogprob - rollout.logprobs
    ratio = logratio.exp()
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95]))
    return {
        f"kl_{suffix}": ((ratio - 1) - logratio).mean().item(),
        f"ratio_q05_{suffix}": q[0].item(),
        f"ratio_q50_{suffix}": q[1].item(),
        f"ratio_q95_{suffix}": q[2].item(),
        f"clipfrac_{suffix}": ((ratio - 1).abs() > clip_coef).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LOG_FIELDS = [
    "update", "global_step",
    "eval_return_mean", "eval_return_std", "train_return_recent",
    "kl_start", "ratio_q05_start", "ratio_q50_start", "ratio_q95_start", "clipfrac_start",
    "kl_end", "ratio_q05_end", "ratio_q50_end", "ratio_q95_end", "clipfrac_end",
    "pg_loss", "v_loss", "entropy",
]


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    eval_env.reset(seed=args.seed + 10_000)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ac = ActorCritic(obs_dim, act_dim)
    agent = PPOAgent(ac, clip_coef=args.clip_coef, loss_type=args.loss)
    optimizer = optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)

    normalizer = Normalizer(obs_dim, args.gamma)
    runner = EnvRunner(env, args.seed, normalizer, update_stats=True)

    run_name = args.run_name or f"k{args.staleness}_{args.loss}__seed{args.seed}"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, run_name + ".csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    writer.writeheader()

    num_updates = args.total_timesteps // args.rollout_steps
    global_step = 0
    # The k + 1 most recent rollouts; history[0] was collected k iterations ago
    # (or is the oldest available one early in training).
    history = collections.deque(maxlen=args.staleness + 1)
    for update in range(1, num_updates + 1):
        # 1. Collect a fresh rollout with the current policy (same env steps for every k)
        history.append(runner.collect(ac, args.rollout_steps))
        global_step += args.rollout_steps

        # 2. Select the stale rollout, re-evaluate its values with the current critic,
        #    and compute advantages and returns (GAE). Actions and behavior
        #    log-probabilities are kept as collected.
        data = history[0]
        data.refresh_values(ac)
        data.compute_gae(args.gamma, args.gae_lambda)

        row = {"update": update, "global_step": global_step}
        try:
            # 3. Diagnostics before the update: current policy vs. the data-collecting policy
            row.update(ratio_stats(ac, data, args.clip_coef, "start"))

            # 4. Optimization epochs over shuffled mini-batches
            stats = []
            inds = np.arange(args.rollout_steps)
            for _ in range(args.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, args.rollout_steps, args.batch_size):
                    mb = inds[start:start + args.batch_size]
                    stats.append(agent.update(data.obs[mb], data.actions[mb], data.logprobs[mb],
                                              data.advantages[mb], data.returns[mb], optimizer))
            row["pg_loss"], row["v_loss"], row["entropy"] = np.mean(stats, axis=0)

            # 5. Diagnostics after the update
            row.update(ratio_stats(ac, data, args.clip_coef, "end"))
            if not np.isfinite([row["pg_loss"], row["v_loss"], row["kl_end"]]).all():
                raise ValueError("non-finite loss or KL")
        except (ValueError, RuntimeError) as e:
            # Numerical instability (e.g. NaN policy parameters make Normal() raise).
            # Stop here and keep the rows logged so far.
            print(f"Unstable at update {update}, step {global_step}: {e}")
            with open(os.path.join(args.log_dir, run_name + ".unstable"), "w") as f:
                f.write(f"{update},{global_step}\n")
            break
        recent = runner.finished_returns[-10:]
        row["train_return_recent"] = np.mean(recent) if recent else ""
        msg = ""
        if update % args.eval_every == 0 or update == num_updates:
            ev = evaluate_policy(ac, eval_env, args.eval_episodes, normalizer)
            row["eval_return_mean"], row["eval_return_std"] = np.mean(ev), np.std(ev)
            msg = f" | Eval {row['eval_return_mean']:.0f}"
        writer.writerow(row)
        log_file.flush()
        if recent:
            print(f"Update {update}/{num_updates} | Step {global_step} | Train {np.mean(recent):.0f}"
                  f"{msg} | KL end {row['kl_end']:.4f} | Clipfrac end {row['clipfrac_end']:.2f}")

    log_file.close()
    print(f"Logged metrics to {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Basic PPO for Hopper-v5 (ECE 562, Assignment 3)")
    p.add_argument("--env-id", type=str, default="Hopper-v5")
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("-k", "--staleness", type=int, default=0,
                   help="update on the rollout collected k PPO iterations ago")
    p.add_argument("--loss", type=str, default="clip", choices=["clip", "ratio", "naive"],
                   help="surrogate loss: clip (PPO), ratio (unclipped IS), naive (no correction)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=5, help="evaluate every N updates")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
