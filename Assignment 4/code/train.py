"""Train SAC on Hopper-v5.

    python train.py --seed 1                      # default SAC
    python train.py --seed 1 --run-name base      # results go to runs/base/seed1/

Every --eval-every environment steps, one row is appended to
runs/<run-name>/seed<seed>/progress.csv containing the deterministic evaluation
return and the MEAN of every statistic returned by SACAgent.update() since the
previous row. Any new key you add to the dict returned by update() therefore
appears in the CSV automatically.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from replay_buffer import ReplayBuffer
from sac import SACAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Hopper-v5")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-name", default="base")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--buffer-capacity", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def evaluate(env, actor, episodes, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            a = actor.act_deterministic(torch.as_tensor(obs, dtype=torch.float32, device=device)[None])
            obs, r, terminated, truncated, _ = env.step(a[0].cpu().numpy())
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = "cpu"

    run_dir = os.path.join(args.out_dir, args.run_name, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    env.action_space.seed(args.seed)
    eval_env.reset(seed=args.seed + 10_000)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    assert np.allclose(env.action_space.low, -1) and np.allclose(env.action_space.high, 1)

    agent = SACAgent(obs_dim, act_dim, hidden=args.hidden, gamma=args.gamma, tau=args.tau,
                     actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                     alpha_lr=args.alpha_lr, device=device)
    buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_capacity, device=device)

    rows, stats_acc = [], defaultdict(list)
    ep_returns, ep_ret, episodes = [], 0.0, 0
    obs, _ = env.reset(seed=args.seed)
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        # ---------------- act ----------------
        if step <= args.learning_starts:
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
            u = np.arctanh(np.clip(a, -1 + 1e-6, 1 - 1e-6))
            beh_mu = np.zeros(act_dim, np.float32)
            beh_log_std = np.zeros(act_dim, np.float32)
            beh_logp_u = float(-act_dim * np.log(2.0) + np.log(1.0 - np.tanh(u) ** 2).sum())
            is_random = 1
        else:
            with torch.no_grad():
                out = agent.actor.sample(torch.as_tensor(obs, dtype=torch.float32)[None])
            a = out["a"][0].numpy()
            u = out["u"][0].numpy()
            beh_mu = out["mu"][0].numpy()
            beh_log_std = out["log_std"][0].numpy()
            beh_logp_u = out["logp_u"][0].item()
            is_random = 0

        next_obs, r, terminated, truncated, _ = env.step(a)
        ep_ret += r
        # Only true termination stops bootstrapping. Time-limit truncation does not.
        buffer.add(obs, a, u, r, next_obs, terminated,
                   beh_mu, beh_log_std, beh_logp_u, step, is_random)
        obs = next_obs
        if terminated or truncated:
            ep_returns.append(ep_ret)
            episodes += 1
            ep_ret = 0.0
            obs, _ = env.reset()

        # ---------------- learn ----------------
        if step > args.learning_starts:
            batch = buffer.sample(args.batch_size, rng)
            for k, v in agent.update(batch).items():
                stats_acc[k].append(v)

        # ---------------- evaluate & log ----------------
        if step % args.eval_every == 0:
            ev_mean, ev_std = evaluate(eval_env, agent.actor, args.eval_episodes, device)
            row = {"step": step, "eval_return_mean": ev_mean, "eval_return_std": ev_std,
                   "train_return_mean": float(np.mean(ep_returns[-10:])) if ep_returns else np.nan,
                   "episodes": episodes, "time_s": time.time() - t0}
            row.update({k: float(np.mean(v)) for k, v in stats_acc.items()})
            rows.append(row)
            stats_acc = defaultdict(list)
            pd.DataFrame(rows).to_csv(os.path.join(run_dir, "progress.csv"), index=False)
            print(f"[{args.run_name} seed{args.seed}] step {step:>7d}  eval {ev_mean:8.1f}  "
                  f"alpha {row.get('alpha', float('nan')):.3f}  {row['time_s']/60:5.1f} min",
                  flush=True)


if __name__ == "__main__":
    main()
