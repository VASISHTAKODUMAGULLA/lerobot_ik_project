import numpy as np


# Classic (distal) DH:  A_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)
# 1=Pan, 2=Lift, 3=Elbow, 4=Wrist Flex, 5=Wrist Roll, (6=Gripper, not part of the FK chain)
#
# Frame 0 sits on the pan axis at (0.0388353, 0, 0.0624) in the world/base frame, axes parallel to the world.
# Joint angle used in row i:  theta_i = SIGN[i] * q[i] + theta_offset_i   (q = MuJoCo joint value, radians)
#
# At q = 0 the arm is an L-shape (upper arm up, forearm forward), so the lift/elbow/wrist offsets
# are the angles of the links from the +x axis, not zero.
OFF_LIFT  = -np.arctan2(0.11257, 0.0280)    # -76.03 deg, upper arm direction from +x
OFF_ELBOW =  np.arctan2(0.11257, 0.0280) - np.arctan2(0.0052, 0.1349)   # 73.82 deg
OFF_WFLEX = -np.pi / 2 + np.arctan2(0.0052, 0.1349)                     # -87.79 deg
TILT = 0.0486795                            # small wrist roll offset from the URDF (2.8 deg)

DH_PARAMS = [
    # d ,         a ,                       alpha ,    theta_offset
    (0.0542,     0.0303992,                 -np.pi/2,  0.0),         # joint 1 (Pan)
    (-0.0182778, np.hypot(0.0280, 0.11257), 0.0,       OFF_LIFT),    # joint 2 (Lift)
    (0.0,        np.hypot(0.1349, 0.0052),  0.0,       OFF_ELBOW),   # joint 3 (Elbow)
    (0.0181,     0.0,                       np.pi/2,   OFF_WFLEX),   # joint 4 (Wrist Flex)
    (-0.0611,    0.0,                       0.0,       -TILT),       # joint 5 (Wrist Roll)
]

# Pan axis points along -z in MuJoCo, DH wants +z, so joint 1 runs the other way
SIGN = np.array([-1.0, 1.0, 1.0, 1.0, 1.0])

T_BASE_0 = np.eye(4)
T_BASE_0[0:3, 3] = [0.0388353, 0.0, 0.0624]

# End-effector point in the gripper frame (x = up, y = left, z = back at q = 0).
# Zero = the gripper body origin, which is what MuJoCo reports as d.xpos[gripper].
# For a real tool tip use the gripperframe site, e.g. [-0.0079, -0.0002, -0.0981].
TOOL = np.array([0.0, 0.0, 0.0])


def dh_matrix(d, a, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d     ],
        [0,   0,        0,       1     ]
    ], dtype=float)


def forward_kinematics(q):
    # q in radians, MuJoCo joint convention. Only the first 5 values are used, the gripper angle is ignored.
    T = T_BASE_0.copy()

    for i, (d, a, alpha, theta_offset) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(d, a, alpha, SIGN[i] * q[i] + theta_offset)

    pos = T[0:3, 3] + T[0:3, 0:3] @ TOOL
    rot = T[0:3, 0:3]

    return pos, rot