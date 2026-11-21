import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from forward_kinematics import forward_kinematics, T_BASE_0
from inverse_kinematics import inverse_kinematics


import mujoco


model_path = os.path.join(PROJECT_ROOT, "model_files", "scene.xml")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll"
]

qpos_addrs = [
    model.jnt_qposadr[model.joint(n).id]
    for n in JOINT_NAMES
]


# set MuJoCo joint angles
def set_joint_angles(q):
    for addr, qi in zip(qpos_addrs, q):
        data.qpos[addr] = qi

    mujoco.mj_forward(model, data)


# get MuJoCo tool position
def get_mujoco_tool_point():
    j = model.joint("wrist_roll").id

    a = data.xanchor[j]
    u = data.xaxis[j]

    s = data.site_xpos[model.site("gripperframe").id]

    return a + np.dot(s - a, u) * u



targets = {
    "target 1": np.array([0.15, 0.10, 0.10]),
    "target 2": np.array([0.15, -0.10, 0.10]),
    "target 3": np.array([0.20, 0.00, 0.15]),
    "target 4": np.array([0.10, 0.05, 0.20]),
    "target 5": np.array([0.20, 0.05, 0.10]),
}


# initial joint configuration (all zeros)
q = np.zeros(5)

print(f"{'target':<12} {'target [mm]':<25} {'IK [mm]':<25} {'MuJoCo [mm]':<25} {'error (mm)':<12}")
print("-" * 105)


for name, target in targets.items():

    # pan axis is 38.8 mm forward of the MuJoCo base origin
    target_0 = target - T_BASE_0[:3, 3]

    # inverse kinematics
    q = inverse_kinematics(target_0, q)

    # numerical forward kinematics
    pos_fk, _ = forward_kinematics(
        q,
        return_matrix=False,
        in_mujoco_frame=True
    )

    # Compare FK coordinate conventions
    pos_default = forward_kinematics(q)[:3, 3]

    pos_mujoco, _ = forward_kinematics(
        q,
        return_matrix=False,
        in_mujoco_frame=True
    )

    print("default FK:", pos_default)
    print("MuJoCo FK:", pos_mujoco)
    print("difference:", pos_mujoco - pos_default)

    # MuJoCo simulation
    set_joint_angles(q)
    pos_mj = get_mujoco_tool_point()

    # calculate errors
    fk_error = np.linalg.norm(target - pos_fk) * 1000
    mj_error = np.linalg.norm(target - pos_mj) * 1000
    model_error = np.linalg.norm(pos_fk - pos_mj) * 1000

    print(
        f"{name:<12} "
        f"{str(np.round(target * 1000, 1)):<25} "
        f"{str(np.round(pos_fk * 1000, 1)):<25} "
        f"{str(np.round(pos_mj * 1000, 1)):<25} "
        f"{mj_error:<12.2f}"
    )

    print(f"  FK error: {fk_error:.2f} mm | MuJoCo error: {mj_error:.2f} mm | model difference: {model_error:.2f} mm")

    if mj_error < 1.0:
        print("  PASS")
    else:
        print("  FAIL")

    print()
