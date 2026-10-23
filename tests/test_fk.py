import sys
sys.path.append("../src")          # adjust if needed

import numpy as np
import mujoco
import mujoco.viewer
from forward_kinematics import forward_kinematics, fk_pos_rot

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
model = mujoco.MjModel.from_xml_path("/home/nao/lerobot/model_files/scene.xml")
data  = mujoco.MjData(model)

# Joint names in the official model (order is important)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

# Get the qpos addresses for these joints
joint_ids = [model.joint(name).id for name in JOINT_NAMES]
qpos_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

def set_joint_angles(q):
    """Set the robot to the given joint configuration (radians)."""
    q = np.asarray(q, dtype=float).flatten()
    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = q[i] if i < len(q) else 0.0
    mujoco.mj_forward(model, data)   # update kinematics

def get_mujoco_ee_pose():
    """
    Get end-effector position and orientation from MuJoCo.
    We use the body called 'gripper' or the site if available.
    Adjust the body name if needed.
    """
    # Try common body names
    for name in ["gripper", "gripper_link", "moving_jaw_so101_v1", "wrist"]:
        try:
            body_id = model.body(name).id
            pos = data.xpos[body_id].copy()
            # Orientation as rotation matrix
            mat = data.xmat[body_id].reshape(3, 3).copy()
            return pos, mat
        except KeyError:
            continue
    raise RuntimeError("Could not find end-effector body. Print model.body_names to check.")

# -------------------------------------------------
# Test configurations
# -------------------------------------------------
test_configs = {
    "Zero":          np.zeros(6),
    "Stretched":     np.deg2rad([0, -30, 60, 30, 0, 0]),
    "Random 1":      np.deg2rad([45, -20, 40, -10, 90, 0]),
    "Random 2":      np.deg2rad([-30, 10, -50, 20, -45, 0]),
}

print(f"{'Config':<12} {'FK pos':<30} {'MuJoCo pos':<30} {'Error (mm)':<12}")
print("-" * 90)

for name, q in test_configs.items():
    # Your FK
    pos_fk, R_fk = fk_pos_rot(q)

    # MuJoCo
    set_joint_angles(q)
    pos_mj, R_mj = get_mujoco_ee_pose()

    error = np.linalg.norm(pos_fk - pos_mj) * 1000  # mm

    print(f"{name:<12} {np.round(pos_fk,4)}  {np.round(pos_mj,4)}  {error:8.2f}")

print("\nDone. If errors are consistently < 10–15 mm, your FK is good.")
print("Larger systematic errors usually mean DH parameter or frame definition mismatch.")