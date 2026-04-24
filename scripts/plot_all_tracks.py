"""Plot all 6 experiments with dense training-time signals.

Three panels:
  (1) success_once — env/success_once (dense, line) + eval/success_once (sparse, markers)
      x-axis: fraction of training (normalized per run, since PPO updates ≠ SAC grad
      steps ≠ MBPO env steps ≠ GRPO epochs).
  (2) env/return for VLA track on RoboTwin (B4, M1, M2b) — sparse 0/1 × 5 reward.
  (3) env/return for MLP-from-scratch on RoboEval (B1, B2, B3) — dense shaped reward
      with penalties; scale is not comparable to panel 2.

Uses env/return (training rollouts, hundreds of points) instead of eval/return
(~30 points) so the shape of training is visible, not just eval snapshots.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def latest_events(tb_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")))
    if not files:
        files = sorted(glob.glob(os.path.join(tb_dir, "**/events.out.tfevents.*"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No event files in {tb_dir}")
    return files[-1]


def load_scalar(tb_dir: str, tag: str):
    acc = EventAccumulator(latest_events(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return [], []
    evs = acc.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]


def normalize_x(steps):
    if not steps:
        return []
    lo, hi = steps[0], steps[-1]
    if hi == lo:
        return [0.0] * len(steps)
    return [(s - lo) / (hi - lo) for s in steps]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b4", required=True)
    parser.add_argument("--m1", required=True)
    parser.add_argument("--m2b", required=True)
    parser.add_argument("--b1", required=True)
    parser.add_argument("--b2", required=True)
    parser.add_argument("--b3", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    runs = [
        ("B4 zero-shot (RoboTwin)", args.b4, "#888888"),
        ("M1: VLA+GRPO (RoboTwin)", args.m1, "#1f77b4"),
        ("M2b: VLA+GRPO+SE3 (RoboTwin)", args.m2b, "#d62728"),
        ("B1: MLP-PPO (RoboEval)", args.b1, "#2ca02c"),
        ("B2: MLP-SAC (RoboEval)", args.b2, "#9467bd"),
        ("B3: MLP-MBPO (RoboEval)", args.b3, "#ff7f0e"),
    ]

    # ------------------ Panel 1: success_once -------------------------------
    # env/success_once (dense training rollouts, solid line)
    # eval/success_once (sparse, held-out seeds, hollow markers)
    ax = axes[0]
    for name, tb_dir, color in runs:
        env_steps, env_vals = load_scalar(tb_dir, "env/success_once")
        eval_steps, eval_vals = load_scalar(tb_dir, "eval/success_once")

        if name.startswith("B4"):
            if eval_vals:
                ax.axhline(
                    eval_vals[-1], color=color, linestyle="--", linewidth=1.5,
                    label=f"{name} (eval {eval_vals[-1] * 100:.1f}%)",
                )
            continue

        if env_vals:
            ax.plot(
                normalize_x(env_steps), env_vals, "-",
                color=color, linewidth=1.6, alpha=0.9,
                label=f"{name} env (peak {max(env_vals) * 100:.1f}%)",
            )
        if eval_vals:
            # Flag isolated spikes to 1.0 as instrumentation bug (see doc).
            # A spike = eval_success ≥ 0.99 for B1/B3 on RoboEval with return << 0.
            is_roboeval = "(RoboEval)" in name
            x_norm = normalize_x(eval_steps)
            clean_x, clean_y, bug_x, bug_y = [], [], [], []
            for xv, yv in zip(x_norm, eval_vals):
                if is_roboeval and yv >= 0.99:
                    bug_x.append(xv); bug_y.append(yv)
                else:
                    clean_x.append(xv); clean_y.append(yv)
            if clean_x:
                ax.plot(
                    clean_x, clean_y, "o",
                    color=color, markerfacecolor="white", markeredgewidth=1.5,
                    markersize=6, alpha=0.85,
                    label=f"{name} eval (peak {max(clean_y) * 100:.1f}%, n={len(eval_vals)})",
                )
            if bug_x:
                ax.plot(
                    bug_x, bug_y, "x",
                    color=color, markersize=11, markeredgewidth=2.5,
                    label=f"{name} eval=1.0 (instrumentation bug, not real success)",
                )

    ax.set_xlabel("training progress (normalized, 0 → 1)")
    ax.set_ylabel("success_once")
    ax.set_title(
        "Success — env (line, dense) vs eval (markers, sparse)\n"
        "✗ = RoboEval eval=1.0 spikes are an info-key bug (return << 0 at same step), not real success"
    )
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)

    # ------------------ Panel 2: env/return on RoboTwin ---------------------
    ax = axes[1]
    for name, tb_dir, color in runs[:3]:
        steps, vals = load_scalar(tb_dir, "env/return")
        if not vals:
            # B4 zero-shot only has eval/return
            steps, vals = load_scalar(tb_dir, "eval/return")
            if not vals:
                continue
            ax.axhline(
                vals[-1], color=color, linestyle="--", linewidth=1.5,
                label=f"{name} (eval {vals[-1]:.2f})",
            )
            continue
        ax.plot(
            normalize_x(steps), vals, "-o", color=color, linewidth=1.6, markersize=5,
            label=f"{name} env (peak {max(vals):.2f})",
        )

    ax.set_xlabel("training progress (normalized)")
    ax.set_ylabel("env/return  (sparse 0/1 × reward_coef=5)")
    ax.set_title("Return — VLA track (RoboTwin, sparse reward)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # ------------------ Panel 3: env/return on RoboEval ---------------------
    ax = axes[2]
    for name, tb_dir, color in runs[3:]:
        steps, vals = load_scalar(tb_dir, "env/return")
        if not vals:
            continue
        ax.plot(
            normalize_x(steps), vals, "-", color=color, linewidth=1.2, alpha=0.85,
            label=f"{name} env (last {vals[-1]:.1f}, peak {max(vals):.1f})",
        )

    ax.set_xlabel("training progress (normalized)")
    ax.set_ylabel("env/return  (dense shaped reward w/ penalties)")
    ax.set_title("Return — MLP-from-scratch track (RoboEval, dense reward)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Plan 1 full results — env (training rollouts, dense) overlaid with eval "
        "(held-out seeds, sparse).  Return scales differ across envs; panels split.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
