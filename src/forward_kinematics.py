"""
Forward Kinematics for the LeRobot SO-101 (standard DH convention).

    T_{i-1,i} = Rot_z(theta_i) Trans_z(d_i) Trans_x(a_i) Rot_x(alpha_i)

Frames (see report): {0} on the table directly below the shoulder-pan axis,
x0 forward, y0 left, z0 up.  {6} = tool centre point on the wrist-roll axis,
centred between the jaws.

Joint readings q (LeRobot / MuJoCo convention, q = 0: upper arm ~vertical,
forearm + gripper horizontal) are converted to DH angles with
    theta_i = SIGN_i * q_i + OFFSET_i
"""

import numpy as np
from typing import Union, Tuple

# ---------------------------------------------------------------------------
# DH parameters (metres, radians)
# ---------------------------------------------------------------------------
#            d       a       alpha
DH_PARAMS = [
    (0.1166, 0.0304, -np.pi / 2),   # 1 shoulder pan
    (0.0,    0.1160,  0.0),         # 2 shoulder lift
    (0.0,    0.1350,  0.0),         # 3 elbow flex
    (0.0,    0.0,    -np.pi / 2),   # 4 wrist flex
    (0.0611, 0.0,     0.0),         # 5 wrist roll
    (0.0981, 0.0,     0.0),         # 6 tool centre point (fixed, no joint)
]

# theta_i = SIGN_i * q_i + OFFSET_i   (row 6 is fixed: theta_6 = 0)
SIGN = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, 0.0])
OFFSET = np.deg2rad([0.0, -76.03, 73.82, -87.79, 180.0, 0.0])

# Frame {0} relative to the MuJoCo / URDF base origin: pan axis is 38.8 mm forward
T_BASE_0 = np.eye(4)
T_BASE_0[0, 3] = 0.0388

JOINT_LIMITS = np.array([[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69],
                         [-1.65806, 1.65806], [-2.74385, 2.84121]])


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


def joint_to_dh(q: np.ndarray) -> np.ndarray:
    """Joint readings q (5 arm joints, radians) -> 6 DH angles theta."""
    q = np.asarray(q, dtype=float).flatten()[:5]          # a 6th value (gripper) is ignored
    if q.size < 5:
        q = np.pad(q, (0, 5 - q.size))
    return SIGN * np.append(q, 0.0) + OFFSET


def dh_to_joint(theta: np.ndarray) -> np.ndarray:
    """Inverse of joint_to_dh for the 5 arm joints (use this after IK)."""
    theta = np.asarray(theta, dtype=float).flatten()[:5]
    return (theta - OFFSET[:5]) / SIGN[:5]


def forward_kinematics(
    q: np.ndarray,
    return_matrix: bool = True,
    in_mujoco_frame: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    End-effector pose T0e from the joint readings.

    q : array-like, 5 (or 6, gripper ignored) joint angles in radians,
        order [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll(, gripper)]
    return_matrix : True -> 4x4 T0e ; False -> (position (3,), rotation (3,3))
    in_mujoco_frame : True -> pose expressed in the MuJoCo/URDF base frame instead of {0}
    """
    theta = joint_to_dh(q)
    T = T_BASE_0.copy() if in_mujoco_frame else np.eye(4)
    for (d, a, alpha), th in zip(DH_PARAMS, theta):
        T = T @ dh_matrix(a, alpha, d, th)

    if return_matrix:
        return T
    return T[:3, 3].copy(), T[:3, :3].copy()


# ---------------------------------------------------------------------------
# Convenience aliases that match the project specification
# ---------------------------------------------------------------------------
def fk(q: np.ndarray) -> np.ndarray:
    """T = forward_kinematics(q)"""
    return forward_kinematics(q, return_matrix=True)


def fk_pos_rot(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """position, rotation = forward_kinematics(q)"""
    return forward_kinematics(q, return_matrix=False)
