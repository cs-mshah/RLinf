"""SE(3)/SO(3) reward wrapper for RoboTwin lift_pot.

See docs/rob831-project/specs/2026-04-21-robotwin-liftpot-vla-rl-design.md §6.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.robotwin.se3_math import se3_log_map, so3_geodesic_distance

__all__ = ["LiftPotSE3RewardWrapper", "compute_se3_reward_from_info"]


_POSE_FIELDS_3 = ("pot_pos", "target_pos")
_POSE_FIELDS_ROT = ("pot_rot_mat", "target_rot_mat")
_POSE_FIELDS_GRIPPER = (
    "left_gripper_pose",
    "right_gripper_pose",
    "left_handle_pose",
    "right_handle_pose",
)


def compute_se3_reward_from_info(
    info: dict,
    prev_action: Optional[np.ndarray],
    action: np.ndarray,
    weights: dict,
) -> float:
    """Reward function equivalent to LiftPotSE3RewardWrapper.step, as a pure function.

    Use when the underlying env is vectorized across subprocess workers and a
    ``gym.Wrapper`` can't be applied (Case B in plan §13). Populates SE(3)/SO(3)
    diagnostics back into ``info`` in place.

    Missing pose fields fall back to identity/zero: the returned reward is still
    finite, but the corresponding error term contributes zero. The caller should
    check ``info.get("pose_fields_missing")`` on the first step to detect this.
    """
    # Track which expected keys are actually present (for debugging).
    missing = [k for k in _POSE_FIELDS_3 + _POSE_FIELDS_ROT + _POSE_FIELDS_GRIPPER
               if k not in info]
    if missing:
        info.setdefault("pose_fields_missing", missing)

    pot_pos = np.asarray(info.get("pot_pos", np.zeros(3)))
    target_pos = np.asarray(info.get("target_pos", np.zeros(3)))
    pot_rot = np.asarray(info.get("pot_rot_mat", np.eye(3)))
    target_rot = np.asarray(info.get("target_rot_mat", np.eye(3)))

    pos_err = float(np.linalg.norm(pot_pos - target_pos))
    rot_err = so3_geodesic_distance(pot_rot, target_rot)

    def _twist(handle_key: str, gripper_key: str) -> float:
        H = np.asarray(info.get(handle_key, np.eye(4)))
        G = np.asarray(info.get(gripper_key, np.eye(4)))
        return float(np.linalg.norm(se3_log_map(np.linalg.inv(H) @ G)))

    align_err = _twist("left_handle_pose", "left_gripper_pose") + _twist(
        "right_handle_pose", "right_gripper_pose"
    )

    gl = bool(info.get("grasp_left_success", False))
    gr = bool(info.get("grasp_right_success", False))
    lift = float(info.get("lift_distance", 0.0))
    subtask_progress = 0.5 * (float(gl) + float(gr)) + (0.25 if lift > 0.05 else 0.0)
    success = bool(info.get("success", False))

    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if prev_action is None:
        action_rate = 0.0
    else:
        action_rate = float(np.sum((a - prev_action) ** 2))

    reward = (
        -weights.get("w_p", 1.0) * pos_err
        - weights.get("w_R", 0.3) * rot_err
        - weights.get("w_ga", 0.5) * align_err
        + weights.get("w_lift", 5.0) * max(0.0, lift)
        + weights.get("w_grasp", 2.0) * subtask_progress
        + weights.get("w_success", 10.0) * float(success)
        - weights.get("w_smooth", 0.1) * action_rate
    )

    info.update(
        {
            "pose_error_pos": pos_err,
            "pose_error_rot": rot_err,
            "gripper_handle_alignment": align_err,
            "subtask_progress": subtask_progress,
            "action_rate_penalty": action_rate,
            "reward_se3": reward,
        }
    )
    return reward


class LiftPotSE3RewardWrapper(gym.Wrapper):
    """Replaces the base env's reward with an SE(3)/SO(3)-based composite.

    The underlying env is expected to surface the following keys in ``info``
    on each step (populated by the state-obs plumbing in Task 13):

        pot_pos, pot_rot_mat                    # pot pose
        target_pos, target_rot_mat              # target pose
        left_gripper_pose, right_gripper_pose   # 4x4 SE(3) homogeneous
        left_handle_pose, right_handle_pose    # 4x4 SE(3) homogeneous
        grasp_left_success, grasp_right_success # bool
        lift_distance                          # float, meters
        success                                # bool
    """

    def __init__(
        self,
        env: gym.Env,
        w_p: float = 1.0,
        w_R: float = 0.3,
        w_ga: float = 0.5,
        w_lift: float = 5.0,
        w_grasp: float = 2.0,
        w_success: float = 10.0,
        w_smooth: float = 0.1,
    ):
        super().__init__(env)
        self.w_p = w_p
        self.w_R = w_R
        self.w_ga = w_ga
        self.w_lift = w_lift
        self.w_grasp = w_grasp
        self.w_success = w_success
        self.w_smooth = w_smooth
        self._prev_action: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Pose errors on the pot.
        pos_err = float(
            np.linalg.norm(np.asarray(info["pot_pos"]) - np.asarray(info["target_pos"]))
        )
        rot_err = so3_geodesic_distance(info["pot_rot_mat"], info["target_rot_mat"])

        # Gripper-to-handle SE(3) alignment (twist norm).
        xi_l = se3_log_map(
            np.linalg.inv(info["left_handle_pose"]) @ info["left_gripper_pose"]
        )
        xi_r = se3_log_map(
            np.linalg.inv(info["right_handle_pose"]) @ info["right_gripper_pose"]
        )
        align_err = float(np.linalg.norm(xi_l) + np.linalg.norm(xi_r))

        # Task-phase progress (spec §6).
        gl = bool(info.get("grasp_left_success", False))
        gr = bool(info.get("grasp_right_success", False))
        lift = float(info.get("lift_distance", 0.0))
        subtask_progress = 0.5 * (float(gl) + float(gr)) + (0.25 if lift > 0.05 else 0.0)
        success = bool(info.get("success", False))

        # Action smoothness.
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._prev_action is None:
            action_rate = 0.0
        else:
            action_rate = float(np.sum((a - self._prev_action) ** 2))
        self._prev_action = a.copy()

        reward = (
            -self.w_p * pos_err
            - self.w_R * rot_err
            - self.w_ga * align_err
            + self.w_lift * max(0.0, lift)
            + self.w_grasp * subtask_progress
            + self.w_success * float(success)
            - self.w_smooth * action_rate
        )

        info.update(
            {
                "pose_error_pos": pos_err,
                "pose_error_rot": rot_err,
                "gripper_handle_alignment": align_err,
                "subtask_progress": subtask_progress,
                "action_rate_penalty": action_rate,
                "reward_se3": reward,
            }
        )
        return obs, reward, terminated, truncated, info
