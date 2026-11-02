"""Sanity checks for your TODO implementations.

    python test_sac.py

Passing these tests does not guarantee that your implementation is correct (the training
curve in Q4 is the real test), but failing one means something is wrong.
"""

import copy
import sys
import traceback

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from networks import tanh_log_det_jacobian
from sac import SACAgent

OBS, ACT, B = 11, 3, 64
TOL = 1e-5


def make_agent(seed=0):
    torch.manual_seed(seed)
    return SACAgent(OBS, ACT, hidden=32)


def make_batch(seed=0, terminated=None):
    g = torch.Generator().manual_seed(seed)
    term = (torch.rand(B, generator=g) < 0.3).float() if terminated is None \
        else torch.full((B,), float(terminated))
    return {
        "obs": torch.randn(B, OBS, generator=g),
        "act": 0.99 * (2 * torch.rand(B, ACT, generator=g) - 1),
        "rew": torch.randn(B, generator=g),
        "next_obs": torch.randn(B, OBS, generator=g),
        "terminated": term,
    }


def record_samples(agent):
    """Wrap agent.actor.sample so that every output is recorded."""
    calls = []
    original = agent.actor.sample

    def wrapped(obs):
        out = original(obs)
        calls.append(out)
        return out

    agent.actor.sample = wrapped
    return calls


def close(x, y, tol=TOL):
    return torch.allclose(torch.as_tensor(x), torch.as_tensor(y), atol=tol, rtol=tol)


# ---------------------------------------------------------------- TODO 1
def test_actor_sample():
    agent = make_agent()
    obs = make_batch()["obs"]
    out = agent.actor.sample(obs)
    for k, shape in {"a": (B, ACT), "u": (B, ACT), "logp_a": (B,), "logp_u": (B,)}.items():
        assert out[k].shape == shape, f"{k} has shape {tuple(out[k].shape)}, expected {shape}"
    assert (out["a"].abs() < 1).all(), "actions must lie in (-1, 1)"
    assert close(out["a"], torch.tanh(out["u"])), "a must equal tanh(u)"

    mu, log_std = agent.actor.dist_params(obs)
    ref_logp_u = Normal(mu, log_std.exp()).log_prob(out["u"]).sum(-1)
    assert close(out["logp_u"], ref_logp_u), "logp_u is not the pre-tanh Gaussian log-density of u"
    ref_logp_a = ref_logp_u - tanh_log_det_jacobian(out["u"])
    assert close(out["logp_a"], ref_logp_a), "logp_a is missing (or has the wrong) tanh correction"

    out["u"].sum().backward()
    g = agent.actor.mu_head.weight.grad
    assert g is not None and g.abs().sum() > 0, "no gradient through u: use rsample()"

    # Distribution check: many samples at one state.
    one = obs[:1].repeat(20000, 1)
    with torch.no_grad():
        u = agent.actor.sample(one)["u"]
        mu1, ls1 = agent.actor.dist_params(obs[:1])
    assert close(u.mean(0), mu1[0], tol=0.05), "sample mean of u does not match mu"
    assert close(u.std(0), ls1[0].exp(), tol=0.05), "sample std of u does not match sigma"


# ---------------------------------------------------------------- TODO 2
def test_critic_loss_terminal():
    agent = make_agent()
    batch = make_batch(terminated=1.0)
    loss, q1_pred = agent.critic_loss(batch)
    q1 = agent.q1(batch["obs"], batch["act"])
    q2 = agent.q2(batch["obs"], batch["act"])
    ref = 0.5 * (F.mse_loss(q1, batch["rew"]) + F.mse_loss(q2, batch["rew"]))
    assert loss.dim() == 0, "loss must be a scalar"
    assert close(loss, ref), "with terminated = 1 the target must equal r"
    assert close(q1_pred, q1), "q1_pred must be Q_phi1(s, a) on the stored actions"


def test_critic_loss_bootstrap():
    agent = make_agent()
    with torch.no_grad():
        agent.log_alpha.fill_(torch.tensor(0.3).log())
        for p in agent.q2_target.parameters():  # make the two target critics differ
            p.add_(0.1 * torch.randn_like(p))
    batch = make_batch(seed=1)
    calls = record_samples(agent)
    loss, _ = agent.critic_loss(batch)
    assert len(calls) == 1, f"call self.actor.sample exactly once (called {len(calls)} times)"
    nxt = calls[0]
    with torch.no_grad():
        q_next = torch.min(agent.q1_target(batch["next_obs"], nxt["a"]),
                           agent.q2_target(batch["next_obs"], nxt["a"]))
        y = batch["rew"] + agent.gamma * (1 - batch["terminated"]) * (q_next - 0.3 * nxt["logp_a"])
    q1 = agent.q1(batch["obs"], batch["act"])
    q2 = agent.q2(batch["obs"], batch["act"])
    ref = 0.5 * (F.mse_loss(q1, y) + F.mse_loss(q2, y))
    assert close(loss, ref), "soft Bellman target or loss is incorrect"

    loss.backward()
    assert all(p.grad is None for p in agent.actor.parameters()), \
        "the target must not propagate gradients into the actor (use torch.no_grad())"
    assert agent.log_alpha.grad is None, "the critic loss must not propagate gradients into alpha"
    assert all(p.grad is not None for p in agent.q1.parameters()) and \
        all(p.grad is not None for p in agent.q2.parameters()), "both critics must receive gradients"


# ---------------------------------------------------------------- TODO 3
def test_actor_loss():
    agent = make_agent()
    with torch.no_grad():
        agent.log_alpha.fill_(torch.tensor(0.2).log())
        for p in agent.q2.parameters():
            p.add_(torch.randn_like(p))
    obs = make_batch(seed=2)["obs"]
    calls = record_samples(agent)
    loss, logp_a = agent.actor_loss(obs)
    assert len(calls) == 1, f"call self.actor.sample exactly once (called {len(calls)} times)"
    pi = calls[0]
    q1v, q2v = agent.q1(obs, pi["a"]), agent.q2(obs, pi["a"])
    assert (q1v < q2v).any() and (q2v < q1v).any(), "test setup: both critics should be selected"
    q = torch.min(q1v, q2v)
    ref = (0.2 * pi["logp_a"] - q).mean()
    assert loss.dim() == 0, "loss must be a scalar"
    assert close(loss, ref), "actor loss is incorrect (online critics, min over the two)"
    assert close(logp_a, pi["logp_a"]), "return logp_a of the fresh actions"
    loss.backward()
    assert all(p.grad is not None for p in agent.actor.parameters()), \
        "the actor must receive gradients (reparameterization)"
    assert agent.log_alpha.grad is None, "alpha must be treated as a constant in the actor loss"


# ---------------------------------------------------------------- TODO 4
def test_alpha_loss():
    agent = make_agent()
    with torch.no_grad():
        agent.log_alpha.fill_(torch.tensor(0.5).log())
    logp = torch.randn(B, requires_grad=True)
    loss = agent.alpha_loss(logp)
    ref = -(agent.log_alpha.detach() * (logp.detach() + agent.target_entropy)).mean()
    assert loss.dim() == 0, "loss must be a scalar"
    assert close(loss, ref), "temperature loss is incorrect"
    loss.backward()
    assert logp.grad is None, "logp_a must be treated as a constant (detach it)"
    assert close(agent.log_alpha.grad, -(logp.detach() + agent.target_entropy).mean()), \
        "gradient with respect to log_alpha is incorrect"


def test_soft_update():
    agent = make_agent()
    with torch.no_grad():
        for p in list(agent.q1.parameters()) + list(agent.q2.parameters()):
            p.add_(torch.randn_like(p))
    online = [p.clone() for p in list(agent.q1.parameters()) + list(agent.q2.parameters())]
    before = [p.clone() for p in list(agent.q1_target.parameters()) + list(agent.q2_target.parameters())]
    agent.tau = 0.25
    agent.soft_update()
    after = list(agent.q1_target.parameters()) + list(agent.q2_target.parameters())
    for o, b, a in zip(online, before, after):
        assert close(a, 0.75 * b + 0.25 * o), "Polyak update is incorrect (check both critics)"


def test_update_runs():
    agent = make_agent()
    batch = make_batch(seed=3)
    for _ in range(5):
        stats = agent.update(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values()), f"non-finite stats {stats}"


TESTS = [
    ("TODO 1  Actor.sample", test_actor_sample),
    ("TODO 2  critic_loss (terminal)", test_critic_loss_terminal),
    ("TODO 2  critic_loss (bootstrap)", test_critic_loss_bootstrap),
    ("TODO 3  actor_loss", test_actor_loss),
    ("TODO 4a alpha_loss", test_alpha_loss),
    ("TODO 4b soft_update", test_soft_update),
    ("        full update step", test_update_runs),
]

if __name__ == "__main__":
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"PASS  {name}")
        except NotImplementedError as e:
            failed += 1
            print(f"TODO  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e}")
            if "-v" in sys.argv:
                traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} tests passed")
    sys.exit(1 if failed else 0)
