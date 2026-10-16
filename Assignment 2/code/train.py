import argparse
import csv
import os

import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from actor_critic import ActorCritic
from buffer import RolloutBuffer
from ppo import PPOAgent


def evaluate_policy(agent_ac, eval_env, num_episodes=5):
    """
    Runs `num_episodes` evaluation episodes with the deterministic policy mean
    (no exploration noise) and returns the per-episode returns.
    """
    device = next(agent_ac.parameters()).device
    returns = []
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        terminated, truncated = False, False
        total_reward = 0.0
        while not (terminated or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent_ac.actor_mean(obs_tensor).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return returns


def evaluate_and_visualize(agent_ac, env_id="InvertedPendulum-v5", seed=42):
    """
    Renders a single evaluation episode using the deterministic policy mean.
    """
    print("\n" + "=" * 50)
    print("Visualizing Trained Policy...")
    print("=" * 50)

    # Initialize the environment in 'human' rendering mode
    eval_env = gym.make(env_id, render_mode="human")
    obs, _ = eval_env.reset(seed=seed)

    device = next(agent_ac.parameters()).device
    terminated, truncated = False, False
    total_eval_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            # Use deterministic mean action for evaluation
            action_mean = agent_ac.actor_mean(obs_tensor)
            action = action_mean.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_eval_reward += reward
        steps += 1

    print(f"Evaluation Episode Finished | Total Steps: {steps} | Total Reward: {total_eval_reward:.1f}")
    eval_env.close()


LOG_FIELDS = [
    "update", "global_step",
    "eval_return_mean", "eval_return_std",
    "train_return_recent",
    "approx_kl", "clipfrac",
    "loss", "pg_loss", "v_loss", "entropy",
]


def train_ppo(
    env_id="InvertedPendulum-v5",
    total_timesteps=120000,
    rollout_steps=2048,
    batch_size=64,
    update_epochs=10,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    seed=1,
    render_at_end=True,
    eval_every=1,
    eval_episodes=5,
    log_dir="runs",
    run_name=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Separate environment for periodic deterministic evaluation
    eval_env = gym.make(env_id)
    eval_env.reset(seed=seed + 10_000)

    ac = ActorCritic(obs_dim, act_dim).to(device)
    agent = PPOAgent(ac, clip_coef=clip_coef)
    optimizer = optim.Adam(ac.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(rollout_steps, obs_dim, act_dim, device)

    # CSV log with one row per update (evaluation columns are empty on
    # updates where no evaluation was run)
    if run_name is None:
        run_name = f"{env_id}__lam{gae_lambda}__clip{clip_coef}__seed{seed}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    writer.writeheader()

    obs, _ = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    # done = 1.0 if `obs` is the first observation of a new episode (i.e. the
    # previous transition terminated or was truncated), else 0.0
    done = 0.0

    num_updates = total_timesteps // rollout_steps
    global_step = 0
    ep_returns, ep_lens = [], []
    current_return, current_len = 0.0, 0

    # Evaluate the initial policy so learning curves start at step 0
    init_eval = evaluate_policy(ac, eval_env, eval_episodes)
    writer.writerow({"update": 0, "global_step": 0,
                     "eval_return_mean": np.mean(init_eval),
                     "eval_return_std": np.std(init_eval)})
    log_file.flush()

    for update in range(1, num_updates + 1):
        buffer.reset()

        # Collect rollouts sequentially from the single environment
        for step in range(rollout_steps):
            global_step += 1
            with torch.no_grad():
                action, logprob, _, value = ac.get_action_and_value(obs.unsqueeze(0))

            cpu_action = action.squeeze(0).cpu().numpy()
            next_obs_np, reward, terminated, truncated, _ = env.step(cpu_action)
            done_flag = float(terminated or truncated)

            current_return += reward
            current_len += 1

            # dones[t] stores d_t: whether obs[t] starts a new episode. With this
            # convention, dones[t + 1] is the termination flag d_{t+1} used in the
            # GAE recursion, and `done` after the loop is d_T (next_done).
            buffer.insert(
                obs,
                action.squeeze(0),
                logprob.squeeze(0),
                reward,
                done,
                value
            )

            if terminated or truncated:
                ep_returns.append(current_return)
                ep_lens.append(current_len)
                current_return, current_len = 0.0, 0
                next_obs_np, _ = env.reset()

            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            done = done_flag

        # Bootstrap value for unfinished trajectories
        with torch.no_grad():
            next_value = ac.get_value(obs.unsqueeze(0)).squeeze()

        # Compute GAE and returns
        buffer.compute_gae(next_value, done, gamma, gae_lambda)

        # Flatten batch data for mini-batch SGD
        b_obs = buffer.obs
        b_actions = buffer.actions
        b_logprobs = buffer.logprobs
        b_advantages = buffer.advantages
        b_returns = buffer.returns

        # Optimization epochs
        b_inds = np.arange(rollout_steps)
        stats = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, rollout_steps, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                stats.append(agent.update(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    optimizer
                ))
        mean_loss, mean_pg, mean_v, mean_ent = np.mean(stats, axis=0)

        # Approximate KL between the rollout policy (pi_old) and the updated
        # policy, estimated on the full rollout batch:
        #   KL(pi_old || pi_new) ~= E[(r - 1) - log r],  r = pi_new / pi_old
        with torch.no_grad():
            _, new_logprobs, _, _ = ac.get_action_and_value(b_obs, b_actions)
            logratio = new_logprobs - b_logprobs
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean().item()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

        row = {
            "update": update, "global_step": global_step,
            "train_return_recent": np.mean(ep_returns[-10:]) if ep_returns else "",
            "approx_kl": approx_kl, "clipfrac": clipfrac,
            "loss": mean_loss, "pg_loss": mean_pg, "v_loss": mean_v, "entropy": mean_ent,
        }

        # Periodic deterministic evaluation (always evaluate the final policy)
        msg = ""
        if update % eval_every == 0 or update == num_updates:
            eval_returns = evaluate_policy(ac, eval_env, eval_episodes)
            row["eval_return_mean"] = np.mean(eval_returns)
            row["eval_return_std"] = np.std(eval_returns)
            msg = f" | Eval Return: {row['eval_return_mean']:.1f}"

        writer.writerow(row)
        log_file.flush()

        if len(ep_returns) > 0:
            mean_ret = np.mean(ep_returns[-10:])
            print(f"Update {update}/{num_updates} | Step {global_step} | Recent Return: {mean_ret:.1f}"
                  f"{msg} | Approx KL: {approx_kl:.4f}")

    env.close()
    eval_env.close()
    log_file.close()
    print(f"\nLogged metrics to {log_path}")

    # Visualization after training completion
    if render_at_end:
        evaluate_and_visualize(ac, env_id=env_id)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training harness (ECE 562, Assignment 2)")
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v5")
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="PPO clipping epsilon; pass 'inf' (or a large value such as 1000) to disable clipping")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1,
                        help="run a deterministic evaluation every N updates")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="number of episodes per evaluation")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None,
                        help="CSV file name (without extension); defaults to env/lambda/clip/seed")
    parser.add_argument("--no-render", action="store_true",
                        help="skip the rendered evaluation episode at the end (use on headless machines)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ppo(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        seed=args.seed,
        render_at_end=not args.no_render,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )