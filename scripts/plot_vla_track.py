"""Plot the VLA-track comparison: B4 (zero-shot) vs M1 (default reward) vs
M2b (SE(3) reward), both eval/success_once and eval/return on two subplots.

Usage:
    python scripts/plot_vla_track.py \
        --b4 /ocean/.../results/b4_vla_zeroshot_3r_40140113/tensorboard \
        --m1 /ocean/.../results/m1_vla_grpo_40143252/tensorboard \
        --m2b /ocean/.../results/m2b_vla_grpo_se3_40143253/tensorboard \
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
    """Returns (steps, values) for the given tag, or ([], []) if absent."""
    acc = EventAccumulator(latest_events(tb_dir))
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return [], []
    evs = acc.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b4", required=True, help="B4 tensorboard dir")
    parser.add_argument("--m1", required=True, help="M1 tensorboard dir")
    parser.add_argument("--m2b", required=True, help="M2b tensorboard dir")
    parser.add_argument("--out", required=True, help="output PNG path")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # --- Eval success_once ---
    ax = axes[0]
    b4_steps, b4_succ = load_scalar(args.b4, "eval/success_once")
    m1_steps, m1_succ = load_scalar(args.m1, "eval/success_once")
    m2b_steps, m2b_succ = load_scalar(args.m2b, "eval/success_once")

    # B4 as horizontal line (only one eval snapshot, no training dynamics).
    if b4_succ:
        ax.axhline(b4_succ[-1], color="#888888", linestyle="--", linewidth=1.4,
                   label=f"B4 zero-shot (no RL) — {b4_succ[-1] * 100:.2f}%")
    if m1_steps:
        ax.plot(m1_steps, m1_succ, "o-", color="#1f77b4", linewidth=1.8,
                markersize=7, label=f"M1: VLA + GRPO (default reward) — peak {max(m1_succ) * 100:.2f}%")
    if m2b_steps:
        ax.plot(m2b_steps, m2b_succ, "s-", color="#d62728", linewidth=1.8,
                markersize=7, label=f"M2b: VLA + GRPO + SE(3) reward — peak {max(m2b_succ) * 100:.2f}%")

    ax.set_xlabel("training epoch (eval checkpoint)")
    ax.set_ylabel("eval / success_once")
    ax.set_title("Eval success rate")
    ax.set_ylim(-0.02, 0.22)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # --- Eval return ---
    ax = axes[1]
    b4_steps, b4_ret = load_scalar(args.b4, "eval/return")
    m1_steps, m1_ret = load_scalar(args.m1, "eval/return")
    m2b_steps, m2b_ret = load_scalar(args.m2b, "eval/return")

    if b4_ret:
        ax.axhline(b4_ret[-1], color="#888888", linestyle="--", linewidth=1.4,
                   label=f"B4 zero-shot — {b4_ret[-1]:.3f}")
    if m1_steps:
        ax.plot(m1_steps, m1_ret, "o-", color="#1f77b4", linewidth=1.8,
                markersize=7, label=f"M1 peak {max(m1_ret):.3f}")
    if m2b_steps:
        ax.plot(m2b_steps, m2b_ret, "s-", color="#d62728", linewidth=1.8,
                markersize=7, label=f"M2b peak {max(m2b_ret):.3f}")

    ax.set_xlabel("training epoch (eval checkpoint)")
    ax.set_ylabel("eval / return (mean episodic)")
    ax.set_title("Eval mean episodic return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("VLA track on RoboTwin lift_pot: B4 vs M1 (default reward) vs M2b (SE(3) reward)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
