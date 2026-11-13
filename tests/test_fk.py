"""
Validate src/forward_kinematics.py against the MuJoCo model of the SO-101.

    python tests/test_fk.py path/to/SO-ARM100/Simulation/SO101/scene.xml

Both sides compute the SAME point for the same joint angles:
    FK     : tool centre point {6}, expressed in the MuJoCo base frame
    MuJoCo : the point on the wrist-roll axis level with the site 'gripperframe'
             (= tool centre point, centred between the jaws)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import mujoco
from forward_kinematics import forward_kinematics, JOINT_LIMITS

np.set_printoptions(suppress=True)

XML = sys.argv[1] if len(sys.argv) > 1 else "/home/nao/lerobot/model_files/scene.xml"
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
qpos_addrs = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]


def set_joint_angles(q):
    """Set the robot to the given joint configuration (radians)."""
    for addr, qi in zip(qpos_addrs, q):
        data.qpos[addr] = qi
    mujoco.mj_forward(model, data)


def get_mujoco_tool_point():
    """Tool centre point according to MuJoCo: 'gripperframe' site projected onto the
    wrist-roll axis (our DH model places the tool point on that axis)."""
    j = model.joint("wrist_roll").id
    a, u = data.xanchor[j], data.xaxis[j]
    s = data.site_xpos[model.site("gripperframe").id]
    return a + np.dot(s - a, u) * u


test_configs = {
    "Zero":      np.deg2rad([0, 0, 0, 0, 0]),
    "Stretched": np.deg2rad([0, -30, 60, 30, 0]),
    "Random 1":  np.deg2rad([45, -20, 40, -10, 90]),
    "Random 2":  np.deg2rad([-30, 10, -50, 20, -45]),
}

print(f"{'Config':<12} {'FK pos [mm]':<28} {'MuJoCo pos [mm]':<28} {'Error (mm)':<10}")
print("-" * 82)
for name, q in test_configs.items():
    pos_fk, _ = forward_kinematics(q, return_matrix=False, in_mujoco_frame=True)
    set_joint_angles(q)
    pos_mj = get_mujoco_tool_point()
    err = np.linalg.norm(pos_fk - pos_mj) * 1000
    print(f"{name:<12} {str(np.round(pos_fk * 1000, 1)):<28} {str(np.round(pos_mj * 1000, 1)):<28} {err:8.2f}")

rng = np.random.default_rng(0)
errors = []
for q in rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], (1000, 5)):
    pos_fk, _ = forward_kinematics(q, return_matrix=False, in_mujoco_frame=True)
    set_joint_angles(q)
    errors.append(np.linalg.norm(pos_fk - get_mujoco_tool_point()) * 1000)
errors = np.array(errors)
print(f"\n1000 random configurations: max error {errors.max():.2f} mm, mean {errors.mean():.2f} mm")
print("RESULT:", "PASS" if errors.max() < 1.0 else "FAIL", "(limit 1 mm)")
