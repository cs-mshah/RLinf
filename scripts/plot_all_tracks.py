"""Plot all 6 experiments on two panels: success_once for all, return split by env.

B4/M1/M2b are on RoboTwin (sparse 0/1 reward), B1/B2/B3 are on RoboEval (dense shaped
reward) so returns are not directly comparable — panels are split accordingly.
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
    acc = EventAccumulator(latest_events(tb_dir))
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
    parser.add_argument("--b1", required=True)
    parser.add_argument("--b2", required=True)
    parser.add_argument("--b3", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # ------------------ Panel 1: success_once (all 6 runs comparable) ------------------
    ax = axes[0]
    for name, tb_dir, style, color in [
        ("B4 zero-shot (RoboTwin)", args.b4, "--", "#888888"),
        ("M1: VLA+GRPO (RoboTwin)", args.m1, "o-", "#1f77b4"),
        ("M2b: VLA+GRPO+SE3 (RoboTwin)", args.m2b, "s-", "#d62728"),
        ("B1: MLP-PPO (RoboEval)", args.b1, "-", "#2ca02c"),
        ("B2: MLP-SAC (RoboEval)", args.b2, "-", "#9467bd"),
        ("B3: MLP-MBPO (RoboEval)", args.b3, "-", "#ff7f0e"),
    ]:
        try:
            steps, vals = load_scalar(tb_dir, "eval/success_once")
        except FileNotFoundError:
            print(f"{name}: no tfevents, skipping")
            continue
        if not vals:
            continue
        label = f"{name} (peak {max(vals) * 100:.1f}%)"
        if name.startswith("B4"):
            ax.axhline(vals[-1], color=color, linestyle="--", linewidth=1.4, label=label)
        else:
            ax.plot(steps, vals, style, color=color, linewidth=1.5, markersize=5, alpha=0.85, label=label)

    ax.set_xlabel("training checkpoint")
    ax.set_ylabel("eval / success_once")
    ax.set_title("Success rate — all 6 runs")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    # ------------------ Panel 2: return on RoboTwin (B4, M1, M2b) ------------------
    ax = axes[1]
    for name, tb_dir, style, color in [
        ("B4 zero-shot", args.b4, "--", "#888888"),
        ("M1: VLA+GRPO (default reward)", args.m1, "o-", "#1f77b4"),
        ("M2b: VLA+GRPO+SE3 reward", args.m2b, "s-", "#d62728"),
    ]:
        try:
            steps, vals = load_scalar(tb_dir, "eval/return")
        except FileNotFoundError:
            continue
        if not vals:
            continue
        if name.startswith("B4"):
            ax.axhline(vals[-1], color=color, linestyle="--", linewidth=1.4, label=f"{name} ({vals[-1]:.2f})")
        else:
            ax.plot(steps, vals, style, color=color, linewidth=1.5, markersize=5, label=f"{name} (peak {max(vals):.2f})")
    ax.set_xlabel("training checkpoint")
    ax.set_ylabel("eval / return  (sparse 0/1 × 5)")
    ax.set_title("Return — VLA track on RoboTwin")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # ------------------ Panel 3: return on RoboEval (B1, B2, B3) ------------------
    ax = axes[2]
    for name, tb_dir, color in [
        ("B1: MLP-PPO", args.b1, "#2ca02c"),
        ("B2: MLP-SAC", args.b2, "#9467bd"),
        ("B3: MLP-MBPO", args.b3, "#ff7f0e"),
    ]:
        try:
            steps, vals = load_scalar(tb_dir, "eval/return")
        except FileNotFoundError:
            continue
        if not vals:
            continue
        ax.plot(steps, vals, "-", color=color, linewidth=1.5, alpha=0.85, label=f"{name} (last {vals[-1]:.1f})")
    ax.set_xlabel("training checkpoint")
    ax.set_ylabel("eval / return  (dense shaped w/ penalties)")
    ax.set_title("Return — MLP-from-scratch track on RoboEval")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Plan 1 full results: VLA track (RoboTwin, sparse reward) vs MLP-from-scratch (RoboEval, dense reward)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
