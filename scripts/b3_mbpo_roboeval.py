"""B3 — minimal MBPO on RoboEval lift_pot.

Standalone training script (does not use RLinf's SAC worker). Implements:
  - SAC actor + twin Q-nets + target Q
  - Dynamics model: MLP predicting state-delta and reward from (s, a)
  - Dyna-1 style: collect real (s,a,r,s') -> train dynamics -> generate one synthetic (s,a,r_hat,s_hat)
    -> train SAC on 50/50 mix of real and synthetic batches
  - Eval periodically and log to tensorboard

Run:
  python scripts/b3_mbpo_roboeval.py \
    --config examples/embodiment/config/env/roboeval_liftpot_state.yaml \
    --out ../results/b3_mbpo_<SLURM_JOB_ID>/tensorboard
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


# -------- Models --------


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, n_hidden: int = 2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low: torch.Tensor, act_high: torch.Tensor):
        super().__init__()
        self.trunk = MLP(obs_dim, 256, 256, n_hidden=1)
        self.mu_head = nn.Linear(256, act_dim)
        self.log_std_head = nn.Linear(256, act_dim)
        self.register_buffer("act_low", act_low)
        self.register_buffer("act_high", act_high)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.trunk(obs))
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-5, 2)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        std = log_std.exp()
        noise = torch.randn_like(mu)
        z = mu + std * noise
        a_tanh = torch.tanh(z)
        # Gaussian log-prob with tanh correction.
        logp = (-0.5 * ((z - mu) / (std + 1e-8)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log1p(-a_tanh.pow(2) + 1e-6).sum(-1)
        # Scale to action bounds.
        a = self.act_low + (a_tanh + 1.0) * 0.5 * (self.act_high - self.act_low)
        return a, logp

    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self(obs)
        a_tanh = torch.tanh(mu)
        return self.act_low + (a_tanh + 1.0) * 0.5 * (self.act_high - self.act_low)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, 256, n_hidden=2)
        self.q2 = MLP(obs_dim + act_dim, 1, 256, n_hidden=2)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class DynamicsModel(nn.Module):
    """Predicts (state_delta, reward) given (s, a). Deterministic MLP."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, obs_dim + 1, 256, n_hidden=2)

    def forward(self, obs, act):
        out = self.net(torch.cat([obs, act], dim=-1))
        delta = out[..., :-1]
        reward = out[..., -1]
        return delta, reward

    def predict_next(self, obs, act):
        with torch.no_grad():
            delta, r = self(obs, act)
            return obs + delta, r


# -------- Replay Buffer --------


class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, device):
        self.size = size
        self.obs = torch.zeros(size, obs_dim, device=device)
        self.act = torch.zeros(size, act_dim, device=device)
        self.rew = torch.zeros(size, device=device)
        self.next_obs = torch.zeros(size, obs_dim, device=device)
        self.done = torch.zeros(size, device=device)
        self.ptr = 0
        self.count = 0

    def add_batch(self, obs, act, rew, next_obs, done):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.obs.device) + self.ptr) % self.size
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self.ptr = (self.ptr + n) % self.size
        self.count = min(self.count + n, self.size)

    def sample(self, n: int):
        idx = torch.randint(0, self.count, (n,), device=self.obs.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


# -------- Env setup --------


def make_env(cfg_path: str, num_envs: int, seed: int, is_eval: bool, device):
    """Instantiate a RoboEvalEnv given the env config YAML."""
    from rlinf.envs.roboeval.roboeval_env import RoboEvalEnv

    cfg = OmegaConf.load(cfg_path)
    cfg.total_num_envs = num_envs
    cfg.is_eval = is_eval
    cfg.auto_reset = True
    cfg.ignore_terminations = True
    env_path = os.environ.get("REPO_PATH", os.getcwd())
    # Minimal video config defaults.
    if "video_cfg" not in cfg:
        cfg.video_cfg = OmegaConf.create({"save_video": False, "info_on_video": False,
                                          "video_base_dir": "", "video_record_interval": 100})
    return RoboEvalEnv(cfg=cfg, num_envs=num_envs, seed_offset=seed,
                       total_num_processes=1, worker_info=None)


# -------- Training --------


def flatten_obs(obs_dict):
    """RoboEvalEnv returns {'states': tensor [N, D]} in state mode."""
    return obs_dict["states"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="examples/embodiment/config/env/roboeval_liftpot_state.yaml")
    p.add_argument("--out", required=True, help="tensorboard output dir")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--start-steps", type=int, default=2_000, help="random actions before training")
    p.add_argument("--train-envs", type=int, default=16)
    p.add_argument("--eval-envs", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--max-ep-len", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--real-frac", type=float, default=0.5,
                   help="fraction of each SAC batch that is real vs synthetic")
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--dyn-updates-per-step", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))

    # Build envs (train + eval).
    print("[b3] building train env...")
    train_env = make_env(args.config, args.train_envs, seed=args.seed, is_eval=False, device=device)
    print("[b3] building eval env...")
    eval_env = make_env(args.config, args.eval_envs, seed=args.seed + 10000, is_eval=True, device=device)

    # Dimensions.
    obs, _ = train_env.reset()
    obs_t = flatten_obs(obs).to(device)
    obs_dim = obs_t.shape[-1]
    act_low = torch.as_tensor(train_env.single_action_space.low, dtype=torch.float32, device=device)
    act_high = torch.as_tensor(train_env.single_action_space.high, dtype=torch.float32, device=device)
    act_dim = act_low.numel()
    print(f"[b3] obs_dim={obs_dim}, act_dim={act_dim}")

    # Models.
    actor = GaussianActor(obs_dim, act_dim, act_low, act_high).to(device)
    q = TwinQ(obs_dim, act_dim).to(device)
    q_target = TwinQ(obs_dim, act_dim).to(device)
    q_target.load_state_dict(q.state_dict())
    for pp in q_target.parameters():
        pp.requires_grad = False
    dyn = DynamicsModel(obs_dim, act_dim).to(device)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_q = torch.optim.Adam(q.parameters(), lr=args.lr)
    opt_dyn = torch.optim.Adam(dyn.parameters(), lr=args.lr)

    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim, device)

    obs_batch = obs_t
    total_steps = 0
    episode_returns = torch.zeros(args.train_envs, device=device)
    best_eval_success = 0.0
    start_time = time.time()

    print("[b3] starting loop")
    while total_steps < args.total_steps:
        # 1) Collect one env step per train env.
        if total_steps < args.start_steps:
            act_np = np.stack([train_env.single_action_space.sample() for _ in range(args.train_envs)])
            act_t = torch.as_tensor(act_np, dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                act_t, _ = actor.sample(obs_batch)

        next_obs_dict, rew_t, term_t, trunc_t, _ = train_env.step(act_t)
        next_obs_t = flatten_obs(next_obs_dict).to(device)
        rew_t = rew_t.to(device).float()
        done_t = (term_t | trunc_t).to(device).float()

        buf.add_batch(obs_batch, act_t, rew_t, next_obs_t, done_t)
        episode_returns += rew_t
        obs_batch = next_obs_t
        total_steps += args.train_envs

        # 2) Training updates after warm-up.
        if total_steps >= args.start_steps and buf.count >= args.batch_size:
            for _ in range(args.updates_per_step):
                # Train dynamics model.
                for _ in range(args.dyn_updates_per_step):
                    o, a, r, no, d = buf.sample(args.batch_size)
                    delta_pred, r_pred = dyn(o, a)
                    dyn_loss = F.mse_loss(delta_pred, no - o) + F.mse_loss(r_pred, r)
                    opt_dyn.zero_grad()
                    dyn_loss.backward()
                    opt_dyn.step()

                # Sample real batch for SAC.
                o, a, r, no, d = buf.sample(args.batch_size)

                # Generate synthetic transitions from dynamics model (starting from real s,a).
                n_real = int(args.real_frac * args.batch_size)
                n_syn = args.batch_size - n_real
                if n_syn > 0:
                    o_syn = o[:n_syn]
                    # Sample action from current policy for synthetic branch.
                    with torch.no_grad():
                        a_syn, _ = actor.sample(o_syn)
                        no_syn, r_syn = dyn.predict_next(o_syn, a_syn)
                        d_syn = torch.zeros_like(r_syn)
                    o_aug = torch.cat([o[n_real:], o_syn], dim=0) if n_real < args.batch_size else o_syn
                    # Actually replace the synthetic-fraction of the batch.
                    o = torch.cat([o[:n_real], o_syn], dim=0)
                    a = torch.cat([a[:n_real], a_syn], dim=0)
                    r = torch.cat([r[:n_real], r_syn], dim=0)
                    no = torch.cat([no[:n_real], no_syn], dim=0)
                    d = torch.cat([d[:n_real], d_syn], dim=0)

                # Q update.
                with torch.no_grad():
                    na, nlogp = actor.sample(no)
                    nq1, nq2 = q_target(no, na)
                    nq = torch.min(nq1, nq2) - args.alpha * nlogp
                    target_q = r + args.gamma * (1.0 - d) * nq
                q1, q2 = q(o, a)
                q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                opt_q.zero_grad()
                q_loss.backward()
                opt_q.step()

                # Actor update.
                a_pi, logp_pi = actor.sample(o)
                qa1, qa2 = q(o, a_pi)
                qa = torch.min(qa1, qa2)
                actor_loss = (args.alpha * logp_pi - qa).mean()
                opt_actor.zero_grad()
                actor_loss.backward()
                opt_actor.step()

                # Soft target update.
                with torch.no_grad():
                    for p_src, p_tgt in zip(q.parameters(), q_target.parameters()):
                        p_tgt.data.mul_(1.0 - args.tau).add_(args.tau * p_src.data)

            # Log every ~1000 env steps.
            if (total_steps // args.train_envs) % 50 == 0:
                writer.add_scalar("train/q_loss", q_loss.item(), total_steps)
                writer.add_scalar("train/actor_loss", actor_loss.item(), total_steps)
                writer.add_scalar("train/dyn_loss", dyn_loss.item(), total_steps)
                writer.add_scalar("train/alpha", args.alpha, total_steps)

        # Episode bookkeeping: track return when episode ends.
        if done_t.any():
            for i in torch.nonzero(done_t, as_tuple=False).view(-1).tolist():
                writer.add_scalar("env/return", episode_returns[i].item(), total_steps)
                episode_returns[i] = 0.0

        # 3) Periodic eval.
        if total_steps > 0 and (total_steps // args.train_envs) % (args.eval_every // args.train_envs) == 0:
            obs_eval, _ = eval_env.reset()
            obs_eval_t = flatten_obs(obs_eval).to(device)
            ep_ret = torch.zeros(args.eval_envs, device=device)
            success_any = torch.zeros(args.eval_envs, dtype=torch.bool, device=device)
            for step in range(args.max_ep_len):
                with torch.no_grad():
                    act_e = actor.greedy(obs_eval_t)
                obs_eval, rew_e, term_e, trunc_e, info_e = eval_env.step(act_e)
                obs_eval_t = flatten_obs(obs_eval).to(device)
                ep_ret += rew_e.to(device).float()
                if "success" in info_e:
                    success_any = success_any | info_e["success"].bool().to(device)
            mean_success = float(success_any.float().mean().item())
            mean_ret = float(ep_ret.mean().item())
            writer.add_scalar("eval/success_once", mean_success, total_steps)
            writer.add_scalar("eval/return", mean_ret, total_steps)
            elapsed = time.time() - start_time
            print(f"[b3] steps={total_steps:7d} | eval/success={mean_success:.4f} | eval/return={mean_ret:8.2f} | elapsed={elapsed/60:.1f}m")
            if mean_success > best_eval_success:
                best_eval_success = mean_success

    print(f"[b3] done. best eval/success_once={best_eval_success:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
