import numpy as np


# 1=Pan, 2=Lift, 3=Elbow, 4=Wrist Flex, 5=Wrist Roll, (6=Gripper)
"""DH_PARAMS = [
    # d ,  a ,  alpha ,  theta_offset
    (0.1166,           0.0304,        np.pi,           0),         # joint 1 (Pan)
    (0.0000,           0.1160,        0,                 np.pi/2),   # joint 2 (Lift) 
    (0.0000,           0.1349,        0,           0),         # joint 3 (Elbow)
    (0.0000,           0.0000,        np.pi/2,           0),         # joint 4 (Wrist Flex)
    (0.0611 + 0.0981,  0.0000,        np.pi,                 0),         # joint 5 (Wrist Roll) + distance to ee
]"""
DH_PARAMS = [
    # d ,  a ,  alpha ,  theta_offset
    (0.0624,           0.0388353,        0,           0),         # joint 1 (Pan)
    (0.0,              0.06208,       -np.pi/2 ,          0),   # joint 2 (Lift) 
    (0.0,              0.11257,        0,           0),         # joint 3 (Elbow)
    (-0.1349,          0.028,         np.pi/2,           0),         # joint 4 (Wrist Flex)
    (0.0075,           0.0611,        np.pi/2,                 0),         # joint 5 (Wrist Roll) 
    (0.0,              0.095,        0,                 0), 
]

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
    T = np.eye(4)

    for i, joint_angle in enumerate(q):
        d, a, alpha, theta = DH_PARAMS[i]
        T_joint = dh_matrix(d, a, alpha, joint_angle + theta)
        T = T @ T_joint

    pos = T[0:3, 3]
    rot = T[0:3, 0:3]

    return pos, rot