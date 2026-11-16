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

## Assignment 4 submission: what was changed and how to reproduce every figure

### Section 1 (TODOs)
TODO 1 in `networks.py`, TODOs 2, 3, 4a, 4b in `sac.py`. Test output: `test_output.txt`
(`python test_sac.py > test_output.txt`).

### Section 2 changes
| File | Change |
|---|---|
| `sac.py` | `SACAgent.diagnostics()` (Q4a: `data_age_mean`, `kl_beh`, `log_ratio_p05/p50/p95`, `clipfrac_beh`), called at the start of `update()` under `no_grad`. Options `target_action` (Q5a) and `actor_loss_type` (Q5b, logs `clipfrac_actor` for clip). |
| `replay_buffer.py` | Stores `next_act`, `next_beh_logp_a`, `has_next` per transition; `set_next()` fills them in. |
| `train.py` | Flags `--target-action {current,replay}` and `--actor-loss {reparam,naive,clip}`; back-fills the next action of the previous transition in the same episode; passes the current step as `batch["cur_step"]`; stops a run cleanly (writing `diverged.txt`) if `update()` returns non-finite values. |
| `plot.py` | Per-run columns/step columns, `--scale`, `--hline`, `--xmax`; zero values dropped on log axes. |

Defaults are unchanged, so `python train.py` is still standard SAC.

### Commands
```bash
python test_sac.py > test_output.txt

# All 12 runs (4 configurations x seeds 1,2,3) in parallel; same as
#   python train.py --seed S --run-name base
#   python train.py --seed S --run-name replay --target-action replay
#   python train.py --seed S --run-name naive  --actor-loss naive
#   python train.py --seed S --run-name clip   --actor-loss clip
./run_all.sh

# Every figure in the report (written to figs/). Uses runs/ and a3_runs/.
./make_figures.sh
```
`a3_runs/k{0,2,4}/seed{1,2,3}/progress.csv` are copies of the Assignment 3 Q5 PPO runs
(clip loss, lag k = 0, 2, 4), `Assignment 3/codes/runs/k*_clip__seed*.csv`, rearranged into
the folder layout `plot.py` reads. Their step column is `global_step`.
