"""
Forward Kinematics for the LeRobot SO-101
Using the student-provided DH parameters (standard Craig convention).
"""

import numpy as np
from typing import Union, Tuple

# ---------------------------------------------------------------------------
# DH parameters (converted to metres and radians)
# ---------------------------------------------------------------------------
# joint | d (m)  | a (m)  | alpha (rad) | theta
DH_PARAMS = [
    (0.062, 0.000,  np.pi/2, None),   # 1
    (0.000, 0.054,  0.0,     None),   # 2
    (0.000, 0.113,  0.0,     None),   # 3
    (0.000, 0.135,  np.pi/2, None),   # 4
    (0.061, 0.000,  0.0,     None),   # 5
    (0.098, 0.000,  0.0,     None),   # 6  (tool / gripper)
]

def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard DH homogeneous transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d     ],
        [0,   0,        0,       1     ]
    ], dtype=float)

def forward_kinematics(
    q: np.ndarray,
    return_matrix: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the end-effector pose from joint configuration.

    Parameters
    ----------
    q : array-like, shape (6,) or (n,)
        Joint angles in **radians**.
        q = [q1, q2, q3, q4, q5, q6]^T
    return_matrix : bool
        If True  → return the full 4×4 homogeneous matrix T0e
        If False → return (position, rotation) where
                   position : (3,) ndarray
                   rotation : (3,3) ndarray ∈ SO(3)

    Returns
    -------
    T0e or (p, R)
    """
    q = np.asarray(q, dtype=float).flatten()
    if q.size < 6:
        # Pad missing joints with zero (useful when gripper is not used)
        q = np.pad(q, (0, 6 - q.size), constant_values=0.0)
    elif q.size > 6:
        raise ValueError("Expected at most 6 joint angles")

    T = np.eye(4)
    for i, (d, a, alpha, _) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i])

    if return_matrix:
        return T
    else:
        position = T[:3, 3].copy()
        rotation = T[:3, :3].copy()
        return position, rotation

# ---------------------------------------------------------------------------
# Convenience aliases that match the project specification
# ---------------------------------------------------------------------------
def fk(q: np.ndarray) -> np.ndarray:
    """T = forward_kinematics(q)"""
    return forward_kinematics(q, return_matrix=True)

def fk_pos_rot(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """position, rotation = forward_kinematics(q)"""
    return forward_kinematics(q, return_matrix=False)