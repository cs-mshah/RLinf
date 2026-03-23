"""Plot mean return vs env train steps from TensorBoard results directories.

Supports plotting multiple experiments on the same axes.

Usage:
    # Single experiment
    python scripts/plot_return.py --config config.yaml --results results/run_123

    # Multiple experiments (PPO vs SAC)
    python scripts/plot_return.py \
        --config examples/embodiment/config/roboeval_liftpot_ppo_mlp_v2.yaml \
        --results results/ppo_run \
        --config examples/embodiment/config/roboeval_liftpot_sac_mlp.yaml \
        --results results/sac_run \
        --random-return -42.5 \
        --title "LiftPot PPO vs SAC" \
        --output comparison.png

    # With names and random baseline from JSON
    python scripts/plot_return.py \
        --config ppo.yaml --results results/ppo --name PPO \
        --config sac.yaml --results results/sac --name SAC \
        --random-baseline random_baseline.json
"""

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_config(config_path):
    """Load a Hydra-style YAML config, ignoring defaults/interpolations."""
    return OmegaConf.load(config_path)


def load_experiment(config_path, results_dir, name=None):
    """Load one experiment's config + tensorboard data. Returns a dict."""
    cfg = load_config(config_path)

    total_num_envs = cfg.env.train.total_num_envs
    max_steps = cfg.env.train.max_steps_per_rollout_epoch
    steps_per_epoch = total_num_envs * max_steps

    if name is None:
        name = cfg.runner.logger.experiment_name

    tb_dir = os.path.join(results_dir, "tensorboard")
    if not os.path.isdir(tb_dir):
        raise FileNotFoundError(f"No tensorboard/ directory found in {results_dir}")

    event_files = sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {tb_dir}")
    event_file = event_files[-1]

    ea = EventAccumulator(event_file)
    ea.Reload()

    events = ea.Scalars("env/return")
    env_steps = [(e.step + 1) * steps_per_epoch for e in events]
    returns = [e.value for e in events]

    eval_steps, eval_returns = None, None
    if "eval/return" in ea.Tags()["scalars"]:
        eval_events = ea.Scalars("eval/return")
        eval_steps = [(e.step + 1) * steps_per_epoch for e in eval_events]
        eval_returns = [e.value for e in eval_events]

    return {
        "name": name,
        "steps_per_epoch": steps_per_epoch,
        "total_num_envs": total_num_envs,
        "max_steps": max_steps,
        "env_steps": env_steps,
        "returns": returns,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
    }


class ParseExperimentAction(argparse.Action):
    """Custom action that collects repeated --config/--results/--name triples."""

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None) or []
        items.append(values)
        setattr(namespace, self.dest, items)


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean return vs env steps for one or more experiments",
    )
    parser.add_argument(
        "--config", dest="configs", action=ParseExperimentAction, default=[],
        help="Path to experiment YAML config (repeat for each experiment)",
    )
    parser.add_argument(
        "--results", dest="results_dirs", action=ParseExperimentAction, default=[],
        help="Path to results directory (repeat, matched 1:1 with --config)",
    )
    parser.add_argument(
        "--name", dest="names", action=ParseExperimentAction, default=[],
        help="Override experiment name (optional, matched 1:1 with --config)",
    )
    parser.add_argument(
        "--title", default=None,
        help="Override plot title",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <first_results_dir>/mean_return_vs_steps.png)",
    )

    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--random-baseline", default=None, metavar="JSON",
        help="Path to JSON file produced by eval_random_agent.py",
    )
    baseline_group.add_argument(
        "--random-return", type=float, default=None,
        help="Random agent mean return value (plotted as a dotted horizontal line)",
    )

    args = parser.parse_args()

    if len(args.configs) == 0:
        parser.error("at least one --config is required")
    if len(args.configs) != len(args.results_dirs):
        parser.error("each --config must have a matching --results")

    # Pad names list with None
    names = args.names + [None] * (len(args.configs) - len(args.names))

    # Load all experiments
    experiments = []
    for cfg_path, res_dir, name in zip(args.configs, args.results_dirs, names):
        exp = load_experiment(cfg_path, res_dir, name)
        print(f"Loaded: {exp['name']}  "
              f"({exp['total_num_envs']} envs × {exp['max_steps']} steps = "
              f"{exp['steps_per_epoch']} steps/epoch, "
              f"{len(exp['returns'])} data points)")
        experiments.append(exp)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    multi = len(experiments) > 1

    for exp in experiments:
        if not multi:
            ax.plot(
                exp["env_steps"], exp["returns"],
                linewidth=1.5, label=f"{exp['name']} Train",
            )
        if exp["eval_steps"] is not None:
            style = "o-" if not multi else "-"
            label = f"{exp['name']} Eval" if not multi else exp["name"]
            ax.plot(
                exp["eval_steps"], exp["eval_returns"], style,
                linewidth=1.5, markersize=4, label=label,
            )

    # Random baseline
    random_return = args.random_return
    if random_return is None and args.random_baseline is not None:
        with open(args.random_baseline) as f:
            random_return = json.load(f)["mean_return"]
    if random_return is not None:
        ax.axhline(
            y=random_return, color="gray", linestyle=":",
            linewidth=1.5, label=f"Random Agent ({random_return:.2f})",
        )

    ax.set_xlabel("Environment Train Steps")
    ax.set_ylabel("Mean Return")

    if args.title:
        ax.set_title(args.title)
    elif len(experiments) == 1:
        ax.set_title(f"{experiments[0]['name']} — Mean Return vs Environment Steps")
    else:
        ax.set_title("Mean Return vs Environment Steps")

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = args.output or os.path.join(args.results_dirs[0], "mean_return_vs_steps.png")
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, "mean_return_vs_steps.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
