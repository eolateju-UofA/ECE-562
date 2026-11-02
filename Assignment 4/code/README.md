# ECE 562 Assignment 4: SAC starter code

Soft Actor-Critic for `Hopper-v5`. Section 1 of the handout asks you to complete five
TODO sections (TODO 1 in `networks.py`, TODOs 2, 3, 4a and 4b in `sac.py`). Section 2 asks you
to modify the completed code.

## Setup

```bash
pip install torch "gymnasium[mujoco]" numpy pandas matplotlib
```

Tested with Python 3.13, torch 2.x, gymnasium 1.x, mujoco 3.x.

## Files

| File | Contents |
|---|---|
| `networks.py` | Tanh-Gaussian actor (**TODO 1**), Q-network, `tanh_log_det_jacobian` |
| `replay_buffer.py` | FIFO replay buffer that also stores behavior-policy information |
| `sac.py` | `SACAgent`: critic loss (**TODO 2**), actor loss (**TODO 3**), temperature loss (**TODO 4a**), target update (**TODO 4b**) |
| `train.py` | Data collection, evaluation, CSV logging |
| `test_sac.py` | Unit tests for each TODO |
| `plot.py` | Mean ± std across seeds for any logged column |

## Running

```bash
# Check your TODOs first
python test_sac.py

# One run (about 30-60 minutes on one CPU core)
python train.py --seed 1 --run-name base

# All three seeds in parallel
for s in 1 2 3; do python train.py --seed $s --run-name base & done; wait

# Plot evaluation return
python plot.py --runs runs/base --column eval_return_mean --out base_return.png

# Compare with Assignment 3 PPO. Point --runs at a folder containing seed*/<file>.csv, and
# use --step-col if your Assignment 3 CSV names the step column differently.
python plot.py --runs runs/base ../a3/runs/k0 --labels SAC "PPO k=0" --out sac_vs_ppo.png
```

Results are written to `runs/<run-name>/seed<seed>/progress.csv`, one row every 5,000
environment steps. Each row contains the evaluation return and the mean of every statistic
returned by `SACAgent.update()` since the previous row, so any key you add to that dictionary
is logged automatically.
