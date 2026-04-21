"""Unit tests for rlinf.envs.robotwin.lift_pot_reward_wrapper."""

import numpy as np
import pytest
import gymnasium as gym

from rlinf.envs.robotwin.lift_pot_reward_wrapper import LiftPotSE3RewardWrapper


class _MockLiftPotEnv(gym.Env):
    """Minimal mock env: step returns hand-crafted info with all SE(3) fields."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(10,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        c30 = np.cos(np.pi / 6)
        s30 = np.sin(np.pi / 6)
        info = {
            "pot_pos": np.zeros(3),
            "pot_rot_mat": np.eye(3),
            "target_pos": np.array([0.0, 0.0, 0.1]),
            "target_rot_mat": np.array(
                [[c30, -s30, 0.0], [s30, c30, 0.0], [0.0, 0.0, 1.0]]
            ),
            "left_gripper_pose": np.eye(4),
            "right_gripper_pose": np.eye(4),
            "left_handle_pose": np.eye(4),
            "right_handle_pose": np.eye(4),
            "grasp_left_success": True,
            "grasp_right_success": True,
            "lift_distance": 0.06,
            "success": False,
        }
        return np.zeros(10, dtype=np.float32), 0.0, False, False, info


class _MockEnvWithOverrides(_MockLiftPotEnv):
    """Mock env where specific info keys can be overridden per-step (for sign tests)."""

    def __init__(self, overrides):
        super().__init__()
        self._overrides = overrides

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        info.update(self._overrides)
        return obs, rew, term, trunc, info


# ─── structural tests ──────────────────────────────────────────────────


def test_se3_reward_components_are_finite():
    env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    env.reset()
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert np.isfinite(reward)
    assert "pose_error_pos" in info
    assert "pose_error_rot" in info
    assert info["pose_error_pos"] == pytest.approx(0.1, abs=1e-6)
    assert info["pose_error_rot"] == pytest.approx(np.pi / 6, abs=1e-5)


def test_se3_reward_subtask_progress_both_grasps_plus_lift():
    """Spec §6: 0.5*(gl+gr) + 0.25*1[lift>5cm] = 0.5*2 + 0.25 = 1.25."""
    env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["subtask_progress"] == pytest.approx(1.25)


def test_se3_reward_action_smoothness_zero_on_first_step():
    env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    env.reset()
    _, _, _, _, info = env.step(np.ones(14, dtype=np.float32))
    assert info["action_rate_penalty"] == pytest.approx(0.0)


def test_se3_reward_action_smoothness_nonzero_on_action_change():
    env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    env.reset()
    env.step(np.ones(14, dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))
    assert info["action_rate_penalty"] > 0.0


# ─── sign / monotonicity / weight-ablation tests ───────────────────────


def test_se3_reward_monotone_in_position_error():
    """Reward strictly decreases as pot drifts further from target."""
    near_env = LiftPotSE3RewardWrapper(
        _MockEnvWithOverrides(
            {"pot_pos": np.array([0.0, 0.0, 0.09]), "target_pos": np.array([0.0, 0.0, 0.1])}
        )
    )
    far_env = LiftPotSE3RewardWrapper(
        _MockEnvWithOverrides(
            {"pot_pos": np.array([0.0, 0.0, -0.4]), "target_pos": np.array([0.0, 0.0, 0.1])}
        )
    )
    near_env.reset()
    far_env.reset()
    _, r_near, _, _, _ = near_env.step(np.zeros(14, dtype=np.float32))
    _, r_far, _, _, _ = far_env.step(np.zeros(14, dtype=np.float32))
    assert r_near > r_far, (r_near, r_far)


def test_se3_reward_success_bonus_applied_when_success_true():
    no_success_env = LiftPotSE3RewardWrapper(_MockEnvWithOverrides({"success": False}))
    success_env = LiftPotSE3RewardWrapper(_MockEnvWithOverrides({"success": True}))
    no_success_env.reset()
    success_env.reset()
    _, r_no, _, _, _ = no_success_env.step(np.zeros(14, dtype=np.float32))
    _, r_yes, _, _, _ = success_env.step(np.zeros(14, dtype=np.float32))
    # Default w_success = 10.0 → success flips reward up by exactly 10.
    assert r_yes - r_no == pytest.approx(10.0, abs=1e-5)


def test_se3_reward_weight_zero_on_rotation_removes_that_term():
    default_env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    no_rot_env = LiftPotSE3RewardWrapper(_MockLiftPotEnv(), w_R=0.0)
    default_env.reset()
    no_rot_env.reset()
    _, r_def, _, _, _ = default_env.step(np.zeros(14, dtype=np.float32))
    _, r_0, _, _, _ = no_rot_env.step(np.zeros(14, dtype=np.float32))
    # Base rotation error pi/6, default w_R=0.3 → dropping the term raises reward
    # by exactly 0.3 * pi/6.
    assert r_0 - r_def == pytest.approx(0.3 * np.pi / 6, abs=1e-5)
