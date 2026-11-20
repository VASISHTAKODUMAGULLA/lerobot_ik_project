
import numpy as np
from forward_kinematics import DH_PARAMS, SIGN, dh_matrix, joint_to_dh
 
N_JOINTS = 5
 
 
def geometric_jacobian(q):

    T = np.eye(4)
    z_axes = []
    origins = []

    # joint axes and positions
    theta = joint_to_dh(q)

    for i in range(N_JOINTS):
        z_axes.append(T[:3, 2]) 
        origins.append(T[:3, 3])

        d, a, alpha = DH_PARAMS[i]

        T = T @ dh_matrix(a, alpha, d, theta[i])

    # fixed tool transform
    d, a, alpha = DH_PARAMS[5]

    # compute the end-effector position
    T = T @ dh_matrix(a, alpha, d, theta[5])
    p = T[:3, 3]

    # compute the Jacobian
    J = np.zeros((6, N_JOINTS))

    for i in range(N_JOINTS):
        # linear velocity contribution of joint i
        J[:3, i] = SIGN[i] * np.cross(z_axes[i], p - origins[i])
        # angular velocity contribution of joint i
        J[3:, i] = SIGN[i] * z_axes[i]
    
    return J