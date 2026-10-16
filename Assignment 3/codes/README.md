# ECE 562 Assignment 3 (PPO II): basic PPO for Hopper-v5

A complete PPO implementation. Running it as provided trains standard PPO:

```bash
pip install torch "gymnasium[mujoco]" numpy pandas matplotlib
python train.py --seed 1
```

One run (300k environment steps) takes roughly 5 to 10 minutes on one CPU core.

## Files

| File | Contents |
|---|---|
| `actor_critic.py` | Gaussian actor (state-independent log std) and state-value critic |
| `buffer.py` | Rollout storage and GAE |
| `ppo.py` | PPO mini-batch update: clipped surrogate, value loss, gradient clipping |
| `train.py` | Rollout collection, observation/reward normalization, evaluation, CSV logging |

## Command-line flags

| Flag | Default |
|---|---|
| `--seed` | 1 |
| `--total-timesteps` | 300000 |
| `--rollout-steps` | 2048 |
| `--batch-size` | 64 |
| `--update-epochs` | 10 |
| `--lr` | 3e-4 |
| `--gamma` | 0.99 |
| `--gae-lambda` | 0.95 |
| `--clip-coef` | 0.2 |
| `--eval-every` | 5 (updates) |
| `--eval-episodes` | 5 |
| `--log-dir` | `runs` |
| `--run-name` | `k<k>_<loss>__seed<seed>` |
| `-k`, `--staleness` | 0 (update on the rollout collected k PPO iterations ago) |
| `--loss` | `clip` (`clip` = PPO, `ratio` = unclipped importance ratio, `naive` = no correction) |

## Logged columns (`<log-dir>/<run-name>.csv`, one row per update)

| Column | Meaning |
|---|---|
| `global_step` | environment steps so far |
| `eval_return_mean`, `eval_return_std` | return of the deterministic policy mean over 5 episodes (every `--eval-every` updates) |
| `train_return_recent` | mean return of the last 10 training episodes |
| `kl_start`, `kl_end` | approximate KL(behavior policy ‖ current policy) on the update batch, before / after the update |
| `ratio_q05_*`, `ratio_q50_*`, `ratio_q95_*` | 5th, 50th, 95th percentiles of the ratio r = π_θ / π_behavior on the update batch |
| `clipfrac_start`, `clipfrac_end` | fraction of samples with \|r − 1\| > ε |
| `pg_loss`, `v_loss`, `entropy` | averages over the update's mini-batches |

If a run becomes numerically unstable (non-finite loss/KL or NaN policy), training stops,
the CSV keeps the rows up to that point, and `<log-dir>/<run-name>.unstable` records
`update,global_step` where it happened.

## Modifications (Assignment 3)

- **Staleness `k` (Q5a)** – `train.py`: every PPO iteration still collects one fresh rollout, but the
  update uses the rollout collected k iterations earlier (a deque of the k + 1 most recent rollouts;
  the oldest one is used while fewer than k + 1 exist). Before GAE, `RolloutBuffer.refresh_values`
  (`buffer.py`) re-evaluates V(s_t) and the bootstrap V(s_T) with the current critic; stored actions
  and behavior log-probabilities are unchanged. Diagnostics are computed on the rollout used.
- **Surrogate loss (Q6a)** – `ppo.py`: `loss_type` selects `naive` (−E[log π Â]), `ratio` (−E[r Â]) or
  `clip` (PPO).

## Reproducing the report figures

```bash
JOBS=8 bash run_all.sh   # 15 runs: k=0/2/4 with clip, k=2 with naive/ratio; seeds 1,2,3 -> runs/
python plot.py           # -> figs/
```

`run_all.sh` runs, for each seed in {1, 2, 3}:

```bash
python train.py --staleness 0 --loss clip  --seed $SEED --log-dir runs
python train.py --staleness 2 --loss clip  --seed $SEED --log-dir runs
python train.py --staleness 4 --loss clip  --seed $SEED --log-dir runs
python train.py --staleness 2 --loss naive --seed $SEED --log-dir runs
python train.py --staleness 2 --loss ratio --seed $SEED --log-dir runs
```

| Figure | Runs used |
|---|---|
| `figs/q5_eval_return.png`, `figs/q5_kl_start.png`, `figs/q5_clipfrac_start.png` | k ∈ {0, 2, 4}, clip |
| `figs/q6_eval_return.png`, `figs/q6_kl_end.png` | k = 2, naive / ratio / clip |
