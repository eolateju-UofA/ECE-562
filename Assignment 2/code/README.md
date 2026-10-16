# ECE 562: Assignment 2 — Proximal Policy Optimization (PPO) & Continuous Control

This repository contains the codebase and starter templates for **Assignment 2: Proximal Policy Optimization (PPO) & Continuous Control**.

You will implement the algorithmic components of PPO across three files, evaluate your policy on continuous control benchmarks in MuJoCo (`InvertedPendulum-v5`), and conduct empirical ablation studies.

---

## 1. Installation & Environment Setup

We recommend using a Conda or Python virtual environment with **Python 3.10 or 3.11**.

### Step 1: Create and Activate Virtual Environment

Using Conda:
```bash
conda create -n ppo_env python=3.10 -y
conda activate ppo_env
```

Or using `venv`:
```bash
python3.10 -m venv ppo_env
source ppo_env/bin/activate  # On Windows: ppo_env\Scripts\activate
```

### Step 2: Install PyTorch

Install PyTorch according to your platform (CPU is OK for this benchmark):

- **Linux / Windows (CPU):**
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **Linux / Windows (CUDA):**
  ```bash
  pip install torch
  ```
- **macOS (Apple Silicon):**
  ```bash
  pip install torch
  ```

### Step 3: Install MuJoCo and Gymnasium

Install Gymnasium with the MuJoCo physics engine extras and supporting libraries:

```bash
pip install "gymnasium[mujoco]" numpy matplotlib
```

Verify the environment loads correctly:
```bash
python -c "import gymnasium as gym; env = gym.make('InvertedPendulum-v5'); env.reset(); print('MuJoCo initialized successfully!')"
```

*(Note: If running on a headless cluster or remote server without an active X-display, ensure you set `export MUJOCO_GL=egl` or pass the `--no-render` flag when running scripts).*

---

## 2. Project Structure

```text
.
├── actor_critic.py   # Continuous Gaussian policy and critic definitions (TODO 1)
├── buffer.py         # Rollout storage and sequential GAE recursion (TODO 2)
├── ppo.py            # Clipped surrogate objective, value loss, and optimizer step (TODO 3 & 4)
├── train.py          # Training iteration loop, rollout worker, CLI harness, and visualization
└── README.md
```

You are only expected to edit the code blocks demarcated by:
```python
# =====================================================================
# TODO X: ...
# =====================================================================
...
# =====================================================================
```

Do **not** modify class method signatures or change the execution harness in `train.py`.

---

## 3. Tasks Overview

### `actor_critic.py` (TODO 1)
* **Joint Log-Likelihood:** Construct a `torch.distributions.Normal` object using the actor mean $\mu_\theta(s)$ and standard deviations $\sigma = \exp(\log \sigma)$. Compute joint log-probability by summing over the continuous action dimension:
  $$\log \pi_\theta(a \mid s) = \sum_{i=1}^m \log \pi_\theta(a_i \mid s)$$
* **Joint Entropy:** Compute the analytical differential entropy summed across all $m$ action dimensions. Ensure outputs have shape `(batch_size,)`.

### `buffer.py` (TODO 2)
* **GAE Recursion:** Implement backward recursion for Generalized Advantage Estimation iterating backwards from $t = T-1$ down to $t = 0$:
  $$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_{t+1}) - V(s_t)$$
  $$\hat{A}_t = \delta_t + \gamma \lambda (1 - d_{t+1})\hat{A}_{t+1}$$
* **Returns:** Compute target returns $R_t = \hat{A}_t + V(s_t)$.
* **Boundary Conditions:** Correctly bootstrap using `next_value` and `next_done` at terminal boundary step $t = T - 1$.

### `ppo.py` (TODO 3 & 4)
* **Advantage Standardization:** Normalize advantages across the mini-batch: $\hat{A}_{\text{norm}} = \frac{\hat{A} - \mu}{\sigma + 10^{-8}}$.
* **Surrogate Objective:** Compute probability ratio $r_t(\theta) = \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t))$ and clipped surrogate loss:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[ \min\left(r_t(\theta)\hat{A}_{\text{norm}}, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{\text{norm}}\right) \right]$$
* **Loss Combination:** Combine policy loss, MSE value loss, and entropy bonus:
  $$L = -L^{\text{CLIP}} + c_1 L^{\text{VF}} - c_2 \mathcal{H}$$
* **Gradient Clipping:** Apply gradient norm clipping (`torch.nn.utils.clip_grad_norm_`) with threshold $g_{\text{max}} = 0.5$.

---

## 4. Running Experiments & Ablations

All hyperparameters and environment targets can be configured via CLI flags in `train.py`.

### Task 1: Default Inverted Pendulum Benchmark
Train the baseline policy on `InvertedPendulum-v5` for $1.2 \times 10^5$ steps across three seeds:
```bash
python train.py --env-id InvertedPendulum-v5 --total-timesteps 120000 --seed 1
python train.py --env-id InvertedPendulum-v5 --total-timesteps 120000 --seed 2
python train.py --env-id InvertedPendulum-v5 --total-timesteps 120000 --seed 3
```

### Task 2: GAE $\lambda$ Trade-off Ablation
Evaluate convergence and variance differences for $\lambda \in \{0.0, 0.95, 1.0\}$:
```bash
# Extreme 1: One-step TD residual (biased, low variance)
python train.py --env-id InvertedPendulum-v5 --gae-lambda 0.0 --seed 1

# Default: GAE intermediate balance
python train.py --env-id InvertedPendulum-v5 --gae-lambda 0.95 --seed 1

# Extreme 2: Empirical Monte Carlo baseline (unbiased, high variance)
python train.py --env-id InvertedPendulum-v5 --gae-lambda 1.0 --seed 1
```

### Task 3: Policy Clipping Ablation ($\epsilon = \infty$)
Disable surrogate ratio clipping to observe policy collapse:
```bash
python train.py --env-id InvertedPendulum-v5 --clip-coef 1000.0 --seed 1
```

---

## 5. Submission Requirements

Submit a single archive (`.zip`) through Canvas containing:
1. `report.pdf`: 
   - Written solutions for the theoretical derivations in Section 1 (GAE derivation and pessimistic lower bound analysis).
   - Learning curves (mean $\pm$ one standard deviation) and empirical discussion for the benchmark and three ablations in Section 2.
2. `code/`: Completed source files (`actor_critic.py`, `buffer.py`, and `ppo.py`) with all `TODO` sections implemented.
