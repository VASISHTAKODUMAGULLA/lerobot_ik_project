import numpy as np

from forward_kinematics import forward_kinematics, JOINT_LIMITS
from jacobian import geometric_jacobian

# joint limits imported from FK module
Q_MIN = JOINT_LIMITS[:, 0]
Q_MAX = JOINT_LIMITS[:, 1]

def inverse_kinematics(target_position, q):

    # DLS parameters
    damping = 0.01
    max_iterations = 300
    tolerance = 1e-4

    for i in range(max_iterations):

        # forward kinematics according to phase 1
        T = forward_kinematics(q)

        # current end-effector position
        current_position = T[:3, 3]

        # position error
        pos_error = target_position - current_position

        # check convergence
        if np.linalg.norm(pos_error) < tolerance:
            break

        # get geometric Jacobian (position rows only)
        J = geometric_jacobian(q)[:3, :]

        # Damped least squares
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            pos_error
        )

        # Update joint angles
        q = q + dq

        # Apply joint limits
        q = np.clip(q, Q_MIN, Q_MAX)

    return q


if __name__ == "__main__":

    target_position = np.array([0.15, 0.10, 0.10])

    q_start = np.zeros(5)

    q_solution = inverse_kinematics(target_position, q_start)
    
    T = forward_kinematics(q_solution)