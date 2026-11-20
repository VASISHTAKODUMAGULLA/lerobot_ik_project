import numpy as np


# 1=Pan, 2=Lift, 3=Elbow, 4=Wrist Flex, 5=Wrist Roll, (6=Gripper, not part of the FK chain)
# Every joint turns about an axis w that passes through a point p.
# Both are in the world frame (= base frame) with the robot at q = 0.
# At q = 0 the arm is an L-shape: upper arm points up, forearm points forward (+x).
POE_PARAMS = [
    # w ,           p (x, y, z)
    ((0, 0, -1),  (0.0388353, 0.0,        0.0624)),    # joint 1 (Pan)         = base + (0.0388353, 0, 0.0624)
    ((0, 1, 0),   (0.0692345, -0.0182778, 0.1166)),    # joint 2 (Lift)        = joint 1 + (0.0304, -0.0183, 0.0542)
    ((0, 1, 0),   (0.0972345, -0.0182778, 0.22917)),   # joint 3 (Elbow)       = joint 2 + (0.028, 0, 0.11257)
    ((0, 1, 0),   (0.2321345, -0.0182778, 0.23437)),   # joint 4 (Wrist Flex)  = joint 3 + (0.1349, 0, 0.0052)
    ((-1, 0, 0),  (0.2932345, -0.0001778, 0.23437)),   # joint 5 (Wrist Roll)  = joint 4 + (0.0611, 0.0181, 0)
]


# End-effector point in the joint 5 frame (same axes as the gripper body: x = up, y = left, z = back).
# Zero = the gripper body origin, which is what MuJoCo reports as d.xpos[gripper].
# For a real tool tip use [-0.0075, 0.0, -0.095] (0.0075 down, 0.095 forward) or your caliper-measured value.
TOOL = np.array([0.0, 0.0, 0.0])
TILT = 0.0486795                             # small wrist roll offset from the URDF (2.8 deg)


R_HOME = np.array([
    [0,             0,            -1],
    [-np.sin(TILT), np.cos(TILT),  0],
    [np.cos(TILT),  np.sin(TILT),  0],
])


M = np.eye(4)
M[0:3, 0:3] = R_HOME
M[0:3, 3] = np.array(POE_PARAMS[4][1]) + R_HOME @ TOOL



def poe_matrix(w, p, theta):
    # exp of one joint: rotate by theta about axis w through point p
    w = np.array(w, dtype=float)
    p = np.array(p, dtype=float)

    W = np.array([
        [0,     -w[2],  w[1]],
        [w[2],   0,    -w[0]],
        [-w[1],  w[0],  0   ]
    ])
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = (np.eye(3) - R) @ p
    return T


def forward_kinematics(q):
    # q in radians, MuJoCo joint convention. zip stops after the 5 joints, so a 6th (gripper) value is ignored.
    T = np.eye(4)
    for (w, p), joint_angle in zip(POE_PARAMS, q):
        T = T @ poe_matrix(w, p, joint_angle)

    T = T @ M

    pos = T[0:3, 3]
    rot = T[0:3, 0:3]

    return pos, rot