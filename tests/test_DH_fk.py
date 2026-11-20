import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from DH_forward_kinematics import forward_kinematics, TOOL

import mujoco

MARGIN = 0.003        # max allowed position error [m] and rotation error (Frobenius norm)
HOLD_SECONDS = 1.0    # how long each configuration stays on screen in the viewer
NUM_RANDOM = 10       # how many random configurations to test (in addition to the zero and isolated joint positions)
VISUAL_OFFSET = np.array([0.0, 0.15, 0.0])   # draw the DH frame next to the robot, use zeros to draw it on the tool

# 5 arm joints + gripper, in the order the q vectors use
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EE_BODY = "gripper"          # MuJoCo gripper body = joint 5 frame
TOOL_SITE = "gripperframe"   # tool site defined in the MuJoCo model

# MuJoCo arrows point along their local z axis, these turn z onto x and y
ARROW_X = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
ARROW_Y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


def generate_test_configurations(num_joints, num_random):
    configs = {}

    configs["zero pos (q=0)"] = np.zeros(num_joints)

    for i in range(num_joints):
        q_isolate = np.zeros(num_joints)
        q_isolate[i] = 0.5
        configs[f"joint {i + 1} in testing"] = q_isolate

    rng = np.random.default_rng(42)
    for idx in range(num_random):
        configs[f"random pos {idx + 1}"] = rng.uniform(-1.5, 1.5, num_joints)

    return configs


def draw_frame(scn, pos, rot, axis_len=0.1, axis_width=0.005):
    # x = red, y = green, z = blue
    scn.ngeom = 0
    frames = ((ARROW_X, [1, 0, 0, 0.8]), (ARROW_Y, [0, 1, 0, 0.8]), (np.eye(3), [0, 0, 1, 0.8]))
    for local, rgba in frames:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([axis_width, axis_width, axis_len]),
            pos=np.asarray(pos, dtype=float),
            mat=(rot @ local).flatten(),
            rgba=np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1


def get_id(model, obj_type, name):
    # mj_name2id returns -1 for a wrong name, and data.body(-1) then silently gives the last body
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"'{name}' not found in the MuJoCo model")
    return idx


def evaluate(model, data, ids, qpos_adr, q):
    """Set q, run MuJoCo FK and the DH FK, return the comparison."""
    data.qpos[qpos_adr] = q
    mujoco.mj_forward(model, data)

    # MuJoCo gripper body is joint 5, so move it to the tool with the same TOOL offset the DH code uses
    sim_rot = data.xmat[ids["body"]].reshape(3, 3).copy()
    sim_pos = data.xpos[ids["body"]].copy() + sim_rot @ TOOL

    calc_pos, calc_rot = forward_kinematics(q)

    return {
        "sim_pos": sim_pos,
        "sim_rot": sim_rot,
        "calc_pos": calc_pos,
        "calc_rot": calc_rot,
        "pos_error": np.linalg.norm(sim_pos - calc_pos),
        "rot_error": np.linalg.norm(sim_rot - calc_rot),
        # where MuJoCo's tool site sits in the gripper body frame, the TOOL value that would match it
        "site_in_body": sim_rot.T @ (data.site_xpos[ids["site"]] - data.xpos[ids["body"]]),
        "site_dist": np.linalg.norm(data.site_xpos[ids["site"]] - calc_pos),
    }


def report(title, q, r):
    ok = r["pos_error"] < MARGIN and r["rot_error"] < MARGIN
    print(f"\nconfiguration: {title}. q is {np.round(q, 3)}")
    print(f"positional error:   {r['pos_error']:.6f} with simpos {np.round(r['sim_pos'], 5)} "
          f"and calcpos {np.round(r['calc_pos'], 5)}   [{'OK' if ok else 'CHECK'}]")
    print(f"rotational error:   {r['rot_error']:.6f}")
    print(f"distance to MuJoCo {TOOL_SITE} site: {r['site_dist']:.6f}")
    return ok


def fk_validation(use_viewer=True, hold=HOLD_SECONDS):
    model_path = os.path.join(PROJECT_ROOT, "model_files", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    ids = {
        "body": get_id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY),
        "site": get_id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE),
    }
    # qpos index of each joint, so extra bodies in scene.xml do not shift the q order
    qpos_adr = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])

    test_cases = generate_test_configurations(len(JOINT_NAMES), NUM_RANDOM)
    results = []

    def run_case(title, q, viewer=None):
        if viewer is not None:
            with viewer.lock():
                r = evaluate(model, data, ids, qpos_adr, q)
                draw_frame(viewer.user_scn, r["calc_pos"] + VISUAL_OFFSET, r["calc_rot"])
            viewer.sync()
        else:
            r = evaluate(model, data, ids, qpos_adr, q)
        results.append(report(title, q, r))
        if title.startswith("zero"):
            print(f"TOOL for the {TOOL_SITE} site in the gripper frame: {np.round(r['site_in_body'], 4)}"
                  f"   (current TOOL: {np.round(TOOL, 4)})")
        return r

    if use_viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
            for title, q in test_cases.items():
                if not viewer.is_running():
                    break
                run_case(title, q, viewer)
                time.sleep(hold)
    else:
        for title, q in test_cases.items():
            run_case(title, q)

    print(f"\n{sum(results)}/{len(results)} configurations within {MARGIN} of MuJoCo")
    return all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true", help="run the numeric check without the GUI")
    args = parser.parse_args()
    sys.exit(0 if fk_validation(use_viewer=not args.no_viewer) else 1)