"""Unit tests for rlinf.envs.robotwin.se3_math."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.robotwin.se3_math import se3_log_map, so3_geodesic_distance


# ─── so3_geodesic_distance ──────────────────────────────────────────────


def test_so3_geodesic_identity_is_zero():
    R = np.eye(3)
    assert so3_geodesic_distance(R, R) == pytest.approx(0.0, abs=1e-7)


def test_so3_geodesic_90_deg_z_rotation():
    R_target = np.eye(3)
    R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    assert so3_geodesic_distance(R, R_target) == pytest.approx(np.pi / 2, abs=1e-6)


def test_so3_geodesic_180_deg_is_pi():
    R_target = np.eye(3)
    R = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    assert so3_geodesic_distance(R, R_target) == pytest.approx(np.pi, abs=1e-6)


def test_so3_geodesic_is_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(5):
        R1 = Rotation.random(random_state=rng).as_matrix()
        R2 = Rotation.random(random_state=rng).as_matrix()
        d1 = so3_geodesic_distance(R1, R2)
        d2 = so3_geodesic_distance(R2, R1)
        assert d1 == pytest.approx(d2, abs=1e-6)


def test_so3_geodesic_handles_numerical_edge_cases():
    """trace can drift slightly outside expected range due to float precision."""
    R = np.array([[1.0, 1e-10, 0], [-1e-10, 1.0, 0], [0, 0, 1.0]])
    d = so3_geodesic_distance(R, np.eye(3))
    assert np.isfinite(d)
    assert d >= 0.0


# ─── se3_log_map ────────────────────────────────────────────────────────


def _make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def test_se3_log_identity_is_zero():
    T = np.eye(4)
    xi = se3_log_map(T)
    assert xi.shape == (6,)
    assert np.allclose(xi, 0.0, atol=1e-7)


def test_se3_log_pure_translation():
    T = _make_se3(np.eye(3), np.array([0.1, 0.2, -0.3]))
    xi = se3_log_map(T)
    # For R=I, v component equals t.
    assert np.allclose(xi[:3], [0.1, 0.2, -0.3], atol=1e-6)
    assert np.allclose(xi[3:], 0.0, atol=1e-7)


def test_se3_log_pure_rotation_norm_equals_angle():
    R = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    T = _make_se3(R, np.zeros(3))
    xi = se3_log_map(T)
    assert np.linalg.norm(xi[3:]) == pytest.approx(np.pi / 6, abs=1e-6)
