"""Plot mean return vs env train steps from a TensorBoard results directory.

Usage:
    python scripts/plot_return.py <results_dir>
    python scripts/plot_return.py results/test-roboeval-liftpot_12345
"""

import argparse
import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description="Plot mean return vs env steps")
    parser.add_argument(
        "results_dir",
        help="Path to a results directory containing a tensorboard/ subfolder",
    )
    args = parser.parse_args()

    tb_dir = os.path.join(args.results_dir, "tensorboard")
    if not os.path.isdir(tb_dir):
        raise FileNotFoundError(f"No tensorboard/ directory found in {args.results_dir}")

    STEPS_PER_EPOCH = 32 * 200  # total_num_envs * max_steps_per_rollout_epoch

    # Find the latest (largest) event file
    event_files = sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {tb_dir}")
    event_file = event_files[-1]
    print(f"Reading: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    # Extract mean return (training)
    events = ea.Scalars("env/return")
    epochs = [e.step for e in events]
    env_steps = [(ep + 1) * STEPS_PER_EPOCH for ep in epochs]
    returns = [e.value for e in events]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(env_steps, returns, linewidth=1.5, label="Train Mean Return")

    # Overlay eval return if available
    if "eval/return" in ea.Tags()["scalars"]:
        eval_events = ea.Scalars("eval/return")
        eval_steps = [(e.step + 1) * STEPS_PER_EPOCH for e in eval_events]
        eval_returns = [e.value for e in eval_events]
        ax.plot(
            eval_steps, eval_returns, "o-",
            linewidth=1.5, markersize=4, label="Eval Mean Return",
        )

    ax.set_xlabel("Environment Train Steps")
    ax.set_ylabel("Mean Return")
    ax.set_title("LiftPot PPO — Mean Return vs Environment Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(args.results_dir, "mean_return_vs_steps.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
