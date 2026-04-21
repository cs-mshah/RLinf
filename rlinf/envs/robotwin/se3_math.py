"""Lie-group utilities for the LiftPot SE(3)/SO(3) reward wrapper.

See docs/rob831-project/specs/2026-04-21-robotwin-liftpot-vla-rl-design.md §6.
"""

from __future__ import annotations

import numpy as np

__all__ = ["so3_geodesic_distance", "se3_log_map"]


def so3_geodesic_distance(R: np.ndarray, R_target: np.ndarray) -> float:
    """Geodesic distance on SO(3): angle of R^T R_target.

    Returns a scalar in [0, pi]. Clamps the trace to [-1, 1] (via cos_theta) to
    handle floating-point drift in orthogonal matrices.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    R_target = np.asarray(R_target, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R.T @ R_target)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def se3_log_map(T: np.ndarray) -> np.ndarray:
    """SE(3) log map: 4x4 homogeneous transform -> 6-vector (v, omega).

    Returns [vx, vy, vz, wx, wy, wz] where (wx, wy, wz) is the axis-angle
    rotation vector and (vx, vy, vz) is the translational tangent-space
    component. Robust to theta -> 0 via branching.
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        omega = np.zeros(3)
        v = t.copy()
    else:
        omega_hat = (R - R.T) * (theta / (2.0 * np.sin(theta)))
        omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        V_inv = (
            np.eye(3)
            - 0.5 * omega_hat
            + ((1.0 - A / (2.0 * B)) / (theta * theta)) * (omega_hat @ omega_hat)
        )
        v = V_inv @ t

    return np.concatenate([v, omega])
