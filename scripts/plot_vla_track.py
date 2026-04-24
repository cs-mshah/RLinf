"""Plot the VLA-track comparison: B4 (zero-shot) vs M1 (default reward) vs
M2b (SE(3) reward).

Uses dense training-time signals (env/success_once, env/return) with sparse
eval signals (eval/success_once, eval/return) overlaid as hollow markers.
This matches the plot_all_tracks.py style and exposes that M1's training-
rollout success peaks at only 5.5% (vs M2b's 12.5%), so the two runs are
not actually "tied at 12.5%" on the training signal.

Usage:
    python scripts/plot_vla_track.py \\
        --b4 /ocean/.../results/b4_vla_zeroshot_3r_40140113/tensorboard \\
        --m1 /ocean/.../results/m1_vla_grpo_40141452/tensorboard \\
        --m2b /ocean/.../results/m2b_vla_grpo_se3_40143253/tensorboard \\
        --out docs/rob831-project/results/vla_track_success.png
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
        raise FileNotFoundError(f"No event files in {tb_dir}")
    return files[-1]


def load_scalar(tb_dir: str, tag: str):
    acc = EventAccumulator(latest_events(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return [], []
    evs = acc.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b4", required=True)
    parser.add_argument("--m1", required=True)
    parser.add_argument("--m2b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    runs = [
        ("M1: VLA + GRPO (default reward)", args.m1, "#1f77b4", "o"),
        ("M2b: VLA + GRPO + SE(3) reward", args.m2b, "#d62728", "s"),
    ]

    # --- Panel 1: success_once (env dense line + eval markers) ---
    ax = axes[0]
    b4_steps, b4_succ = load_scalar(args.b4, "eval/success_once")
    if b4_succ:
        ax.axhline(
            b4_succ[-1], color="#888888", linestyle="--", linewidth=1.5,
            label=f"B4 zero-shot (no RL) — {b4_succ[-1] * 100:.2f}%",
        )

    for name, tb_dir, color, marker in runs:
        env_steps, env_succ = load_scalar(tb_dir, "env/success_once")
        eval_steps, eval_succ = load_scalar(tb_dir, "eval/success_once")
        if env_steps:
            ax.plot(
                env_steps, env_succ, "-", color=color, linewidth=1.8,
                alpha=0.9,
                label=f"{name} — env peak {max(env_succ) * 100:.1f}%",
            )
        if eval_steps:
            ax.plot(
                eval_steps, eval_succ, marker, color=color,
                markerfacecolor="white", markeredgewidth=1.8,
                markersize=9, alpha=0.85,
                label=f"{name} — eval peak {max(eval_succ) * 100:.1f}% (n={len(eval_succ)})",
            )

    ax.set_xlabel("GRPO epoch")
    ax.set_ylabel("success_once")
    ax.set_title("VLA track success — env (line, dense) vs eval (markers, sparse)")
    ax.set_ylim(-0.02, 0.22)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # --- Panel 2: return (env dense line + eval markers) ---
    ax = axes[1]
    b4_steps, b4_ret = load_scalar(args.b4, "eval/return")
    if b4_ret:
        ax.axhline(
            b4_ret[-1], color="#888888", linestyle="--", linewidth=1.5,
            label=f"B4 zero-shot — {b4_ret[-1]:.3f}",
        )

    for name, tb_dir, color, marker in runs:
        env_steps, env_ret = load_scalar(tb_dir, "env/return")
        eval_steps, eval_ret = load_scalar(tb_dir, "eval/return")
        if env_steps:
            ax.plot(
                env_steps, env_ret, "-", color=color, linewidth=1.8, alpha=0.9,
                label=f"{name} — env peak {max(env_ret):.2f}",
            )
        if eval_steps:
            ax.plot(
                eval_steps, eval_ret, marker, color=color,
                markerfacecolor="white", markeredgewidth=1.8,
                markersize=9, alpha=0.85,
                label=f"{name} — eval peak {max(eval_ret):.2f}",
            )

    ax.set_xlabel("GRPO epoch")
    ax.set_ylabel("return (RoboTwin sparse 0/1 × reward_coef=5)")
    ax.set_title("VLA track return — env (line, dense) vs eval (markers)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "VLA track on RoboTwin lift_pot: B4 zero-shot vs M1 (default reward) vs M2b (SE(3) reward)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
