"""RL experiment: learning to articulate with 52 muscle actuators.

Task ("articulatory reaching"): every 6 control steps (~100 ms) a new mouth
target arrives (lip aperture, mouth width, protrusion, lower-lip tuck, jaw
height), like the gestures of a phoneme sequence. The policy commands muscle
excitations; activations follow first-order muscle dynamics (activation
tau 30 ms, deactivation 60 ms); the face is the differentiable surrogate of
the muscle simulator.

Reward: -(weighted target error) - effort - excitation jerk.

Compared, same policy network and budget of environment steps:
  * PPO            sample-based policy gradient (the robotics RL default)
  * APG            analytic policy gradient: backprop through the differentiable
                   muscle face over the whole episode (model-based)
Output: experiments/rl_articulation.json (learning curves vs env steps and wall clock).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mimetic.face import anatomy as A  # noqa: E402
from mimetic.face.surrogate import sample_activations  # noqa: E402
from mimetic.model.articulation import measures  # noqa: E402
from mimetic.paths import EXPERIMENTS  # noqa: E402
from mimetic.runtime.face_model import load_face_model  # noqa: E402

DT = 1 / 60
TAU_ACT, TAU_DEACT = 0.030, 0.060
T_EP, HOLD = 36, 6
GOAL_W = torch.tensor([4.0, 1.0, 2.0, 2.0, 1.0])


class Env:
    def __init__(self, face, dev, B):
        self.face, self.dev, self.B = face, dev, B
        self.rest_m = self.meas(face.rest[None])

    def meas(self, x):
        m = measures(x)
        jaw = self.face.rest[152, 1] - x[..., 152, 1]
        return torch.stack([m["aperture"], m["width"], m["protrusion"], m["tuck"], jaw], -1)

    def sample_goals(self, g):
        """Reachable targets: measures of random plausible muscle patterns."""
        a = sample_activations(self.B * (T_EP // HOLD), g).to(self.dev)
        with torch.no_grad():
            goals = self.meas(self.face(a)).view(self.B, T_EP // HOLD, 5)
        return goals.repeat_interleave(HOLD, 1)                    # (B, T, 5)

    def step(self, act, u):
        tau = torch.where(u > act, torch.full_like(u, TAU_ACT), torch.full_like(u, TAU_DEACT))
        act = act + (u - act) * (DT / tau).clamp(max=1)
        x = self.face(act)
        return act, self.meas(x)


class Policy(nn.Module):
    def __init__(self, n_act=A.N_ACT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_act + 10 + 1, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                                 nn.Linear(256, n_act))
        self.log_std = nn.Parameter(torch.full((n_act,), -1.5))
        self.v = nn.Sequential(nn.Linear(n_act + 10 + 1, 256), nn.Tanh(), nn.Linear(256, 1))

    def obs(self, act, meas, goal, t):
        return torch.cat([act, (meas - goal) * 5, goal * 2, torch.full_like(act[:, :1], t / T_EP)], -1)

    def mean(self, o):
        return torch.sigmoid(self.net(o) - 2.0)


def reward(meas, goal, u, u_prev):
    err = (((meas - goal) * 10) ** 2 * GOAL_W.to(meas.device)).sum(-1)
    return -err - 0.02 * u.sum(-1) - 0.5 * ((u - u_prev) ** 2).sum(-1), err


def rollout_apg(env, pol, g):
    goals = env.sample_goals(g)
    act = torch.zeros(env.B, A.N_ACT, device=env.dev)
    meas = env.rest_m.expand(env.B, 5)
    u_prev = torch.zeros_like(act)
    R, E = 0, 0
    for t in range(T_EP):
        u = pol.mean(pol.obs(act, meas, goals[:, t], t))
        act, meas = env.step(act, u)
        r, e = reward(meas, goals[:, t], u, u_prev)
        u_prev = u
        R, E = R + r, E + e
    return R.mean() / T_EP, (E / T_EP).mean()


def train_apg(env, pol, budget, g, log, t0):
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    steps = 0
    while steps < budget:
        R, E = rollout_apg(env, pol, g)
        opt.zero_grad(); (-R).backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        steps += env.B * T_EP
        log.append({"env_steps": steps, "goal_err": E.item(), "wall": time.time() - t0})


def train_ppo(env, pol, budget, g, log, t0, epochs=4, clip=0.2, gamma=0.97, lam=0.95):
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    steps = 0
    while steps < budget:
        with torch.no_grad():
            goals = env.sample_goals(g)
            act = torch.zeros(env.B, A.N_ACT, device=env.dev)
            meas = env.rest_m.expand(env.B, 5)
            u_prev = torch.zeros_like(act)
            O, U, LP, Rw, V, Errs = [], [], [], [], [], []
            for t in range(T_EP):
                o = pol.obs(act, meas, goals[:, t], t)
                mu = pol.mean(o)
                dist = torch.distributions.Normal(mu, pol.log_std.exp())
                u = dist.sample().clamp(0, 1)
                act, meas = env.step(act, u)
                r, e = reward(meas, goals[:, t], u, u_prev)
                u_prev = u
                O.append(o); U.append(u); LP.append(dist.log_prob(u).sum(-1)); Rw.append(r); V.append(pol.v(o)[:, 0])
                Errs.append(e)
            V.append(torch.zeros_like(V[0]))
            adv = torch.zeros(env.B, T_EP, device=env.dev)
            last = torch.zeros(env.B, device=env.dev)
            for t in reversed(range(T_EP)):
                delta = Rw[t] + gamma * V[t + 1] - V[t]
                last = delta + gamma * lam * last
                adv[:, t] = last
            ret = adv + torch.stack(V[:-1], 1)
            O, U, LP = torch.stack(O, 1).flatten(0, 1), torch.stack(U, 1).flatten(0, 1), torch.stack(LP, 1).flatten()
            adv, ret = adv.flatten(), ret.flatten()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(epochs):
            perm = torch.randperm(len(O), device=env.dev)
            for k in range(0, len(O), 2048):
                i = perm[k:k + 2048]
                dist = torch.distributions.Normal(pol.mean(O[i]), pol.log_std.exp())
                ratio = (dist.log_prob(U[i]).sum(-1) - LP[i]).exp()
                pl = -torch.min(ratio * adv[i], ratio.clamp(1 - clip, 1 + clip) * adv[i]).mean()
                vl = ((pol.v(O[i])[:, 0] - ret[i]) ** 2).mean()
                opt.zero_grad(); (pl + 0.5 * vl - 0.001 * dist.entropy().sum(-1).mean()).backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        steps += env.B * T_EP
        log.append({"env_steps": steps, "goal_err": torch.stack(Errs).mean().item(), "wall": time.time() - t0})


@torch.no_grad()
def evaluate(env, pol, g):
    goals = env.sample_goals(g)
    act = torch.zeros(env.B, A.N_ACT, device=env.dev)
    meas = env.rest_m.expand(env.B, 5)
    errs = []
    for t in range(T_EP):
        u = pol.mean(pol.obs(act, meas, goals[:, t], t))
        act, meas = env.step(act, u)
        errs.append(((meas - goals[:, t]).abs() * 10)[:, :4].mean(-1))   # mm, lip measures
    return torch.stack(errs[HOLD // 2:]).mean().item()


def main(budget=3_000_000, methods=("apg", "ppo"), out_name="rl_articulation.json"):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    face = load_face_model(dev)
    out = {}
    # Reference: a policy that never moves (error = how far the targets are from rest).
    env0 = Env(face, dev, B=1024)
    goals = env0.sample_goals(torch.Generator().manual_seed(99))
    out["zero_policy_err_mm"] = ((env0.rest_m[None] - goals).abs() * 10)[..., :4].mean().item()
    print(f"zero policy: {out['zero_policy_err_mm']:.3f} mm", flush=True)
    for method in methods:
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        env = Env(face, dev, B=256 if method == "apg" else 512)
        pol = Policy().to(dev)
        log = []
        t0 = time.time()
        (train_apg if method == "apg" else train_ppo)(env, pol, budget, g, log, t0)
        ev = evaluate(Env(face, dev, B=1024), pol, torch.Generator().manual_seed(99))
        out[method] = {"curve": log[::max(1, len(log) // 200)], "final_mean_abs_err_mm": ev,
                       "wall_s": time.time() - t0}
        print(f"{method}: final lip-target error {ev:.3f} mm after {budget} env steps ({time.time() - t0:.0f}s)", flush=True)
    EXPERIMENTS.mkdir(exist_ok=True)
    (EXPERIMENTS / out_name).write_text(json.dumps(out))


if __name__ == "__main__":
    budget = int(float(sys.argv[1])) if len(sys.argv) > 1 else 3_000_000
    methods = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("apg", "ppo")
    main(budget, methods, sys.argv[3] if len(sys.argv) > 3 else "rl_articulation.json")
