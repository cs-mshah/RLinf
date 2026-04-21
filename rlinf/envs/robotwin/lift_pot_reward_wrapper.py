"""SE(3)/SO(3) reward wrapper for RoboTwin lift_pot.

See docs/rob831-project/specs/2026-04-21-robotwin-liftpot-vla-rl-design.md §6.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.robotwin.se3_math import se3_log_map, so3_geodesic_distance

__all__ = ["LiftPotSE3RewardWrapper"]


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
