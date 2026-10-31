import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from POE_forward_kinematics import forward_kinematics, TOOL

import mujoco
import mujoco.viewer
import numpy as np

MARGIN = 0.001

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

    np.random.seed(42)
    for idx in range(num_random):
        q_rand = np.random.uniform(-1.5, 1.5, num_joints)
        configs[f"random pos {idx+1}"] = q_rand

    return configs

def fk_validation():
    model_path = os.path.join(PROJECT_ROOT, "model_files", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    ee_name = "gripper"
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    num_joints = model.nq
    test_cases = generate_test_configurations(num_joints, num_random=10)

    
    with mujoco.viewer.launch_passive(model, data) as viewer:

        
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

        for title, q in test_cases.items():
            if not viewer.is_running():
                break

            print(f"\nconfiguration: {title} in testing. q is {np.round(q,3)}")

            
            data.qpos[:] = q
            mujoco.mj_forward(model, data)

            # MuJoCo gripper body is joint 5, so move it to the tool with the same TOOL offset
            sim_rot = data.body(ee_id).xmat.copy().reshape(3,3)
            sim_pos = data.body(ee_id).xpos.copy() + sim_rot @ TOOL

            
            calc_pos, calc_rot = forward_kinematics(q)

            # Draw the arrows at the tool. Use e.g. [0.0, 0.15, 0.0] to move them aside
            visual_offset = np.array([0.0, 0.0, 0.0])
            shifted_calc_pos = calc_pos + visual_offset

            
            viewer.user_scn.ngeom = 0
            axis_len = 0.1
            axis_width = 0.005

            
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[axis_width, axis_width, axis_len],
                pos=shifted_calc_pos,
                mat=(calc_rot @ ARROW_X).flatten(),
                rgba=[1, 0, 0, 0.8]
            )
            viewer.user_scn.ngeom += 1

            
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[axis_width, axis_width, axis_len],
                pos=shifted_calc_pos,
                mat=(calc_rot @ ARROW_Y).flatten(),
                rgba=[0, 1, 0, 0.8]
            )
            viewer.user_scn.ngeom += 1

            
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[axis_width, axis_width, axis_len],
                pos=shifted_calc_pos,
                mat=calc_rot.flatten(),
                rgba=[0, 0, 1, 0.8]
            )
            viewer.user_scn.ngeom += 1

            viewer.sync()

            
            pos_error = np.linalg.norm(sim_pos - calc_pos)
            rot_error = np.linalg.norm(sim_rot - calc_rot)
            status = "OK" if pos_error < MARGIN and rot_error < MARGIN else "CHECK"
            print(f"positional error:   {pos_error:.6f} with simpos {sim_pos} and calcpos {calc_pos}   [{status}]")
            print(f"rotational error:   {rot_error:.6f}")

            # how far the schematic tool is from the tool site MuJoCo has in the model
            site_dist = np.linalg.norm(data.site_xpos[site_id] - calc_pos)
            print(f"distance to MuJoCo gripperframe site: {site_dist:.6f}")

            time.sleep(10.0)



if __name__ == "__main__":
    fk_validation()