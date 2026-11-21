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
    "Reachable": {
        "target 1": np.array([0.15, 0.10, 0.10]),
        "target 2": np.array([0.15, -0.10, 0.10]),
        "target 3": np.array([0.20, 0.00, 0.15]),
        "target 4": np.array([0.15, 0.05, 0.20]),
        "target 5": np.array([0.20, 0.05, 0.10]),
    },

    "Unreachable": {
        "target 6": np.array([0.80, 0.00, 0.20]),
        "target 7": np.array([0.80, 0.40, 0.20]),
        "target 8": np.array([0.80, -0.40, 0.20]),
        "target 9": np.array([0.20, 0.00, 0.90]),
        "target 10": np.array([-0.80, 0.00, 0.20]),
    }
}



tolerance = 1.0  # mm

print(f"{'target':<12} {'type':<12} {'target [mm]':<25} {'MuJoCo [mm]':<25} {'error (mm)':<12} {'result'}")
print("-" * 110)

for category, category_targets in targets.items():

    for name, target in category_targets.items():

        # initial joint configuration for each test (all zeros)
        q = np.zeros(5)

        # pan axis is 38.8 mm forward of the MuJoCo base origin
        target_0 = target - T_BASE_0[:3, 3]

        # Inverse kinematics
        q = inverse_kinematics(target_0, q)

        # Numerical forward kinematics
        pos_fk, _ = forward_kinematics(
            q,
            return_matrix=False,
            in_mujoco_frame=True
        )

        # MuJoCo simulation
        set_joint_angles(q)
        pos_mj = get_mujoco_tool_point()

        # Calculate errors
        fk_error = np.linalg.norm(target - pos_fk) * 1000
        mj_error = np.linalg.norm(target - pos_mj) * 1000
        model_error = np.linalg.norm(pos_fk - pos_mj) * 1000

        # Evaluate result
        if category == "Reachable":
            result = "PASS" if mj_error < tolerance else "FAIL"
        else:
            result = "EXPECTED" if mj_error >= tolerance else "CHECK"

        print(
            f"{name:<12} "
            f"{category:<12} "
            f"{str(np.round(target * 1000, 1)):<25} "
            f"{str(np.round(pos_mj * 1000, 1)):<25} "
            f"{mj_error:<12.2f} "
            f"{result}"
        )

        print(
            f"  FK error: {fk_error:.2f} mm | "
            f"MuJoCo error: {mj_error:.2f} mm | "
            f"Model difference: {model_error:.2f} mm"
        )

        print()