"""Evaluate a random agent on the LiftPot environment.

Runs N independent episodes with uniformly random actions and reports the
mean episode return.  Results are printed to stdout and optionally saved
to a JSON file for use by plot_return.py (--random-baseline flag).

Usage:
    python scripts/eval_random_agent.py --config <config.yaml> [--num-episodes 3] [--seed 42] [--output random_baseline.json]
"""

import argparse
import json
import sys

import numpy as np
import torch
from omegaconf import OmegaConf


def make_env(cfg):
    """Create a single-env RoboEvalEnv from an experiment config."""
    from rlinf.envs.roboeval.roboeval_env import RoboEvalEnv

    env_cfg = OmegaConf.to_container(cfg.env.train, resolve=False)
    # Force settings appropriate for evaluation
    env_cfg["total_num_envs"] = 1
    env_cfg["group_size"] = 1
    env_cfg["auto_reset"] = False
    env_cfg["ignore_terminations"] = False
    env_cfg["use_rel_reward"] = False
    env_cfg.setdefault("max_episode_steps", 200)

    # Strip video config entirely to avoid unresolved interpolations
    env_cfg.pop("video_cfg", None)

    env = RoboEvalEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    return env


def run_episode(env, max_steps, rng):
    """Run one episode with random actions. Returns the episode return."""
    env.reset()
    episode_return = 0.0

    for _ in range(max_steps):
        # Sample a random action within the action space bounds
        low = env.action_space.low
        high = env.action_space.high
        action = rng.uniform(low, high).astype(np.float32)
        action_t = torch.from_numpy(action).unsqueeze(0)

        _, reward, terminations, truncations, _ = env.step(action_t, auto_reset=False)
        episode_return += reward.item()

        if terminations.any() or truncations.any():
            break

    return episode_return


def main():
    parser = argparse.ArgumentParser(description="Evaluate a random agent on LiftPot")
    parser.add_argument(
        "--config", required=True,
        help="Path to the experiment YAML config",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=3,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save results as JSON (for plot_return.py --random-baseline)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    max_steps = int(OmegaConf.select(cfg, "env.train.max_episode_steps", default=200))

    print(f"Creating LiftPot environment ...")
    env = make_env(cfg)

    rng = np.random.default_rng(args.seed)
    episode_returns = []

    for ep in range(args.num_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        env.seed = ep_seed
        ret = run_episode(env, max_steps, rng)
        episode_returns.append(ret)
        print(f"  Episode {ep + 1}/{args.num_episodes}: return = {ret:.4f}")

    mean_return = float(np.mean(episode_returns))
    std_return = float(np.std(episode_returns))

    print(f"\nRandom agent over {args.num_episodes} episodes:")
    print(f"  Mean return: {mean_return:.4f}")
    print(f"  Std return:  {std_return:.4f}")
    print(f"  Per-episode: {[round(r, 4) for r in episode_returns]}")

    if args.output:
        result = {
            "mean_return": mean_return,
            "std_return": std_return,
            "num_episodes": args.num_episodes,
            "episode_returns": episode_returns,
            "seed": args.seed,
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
