"""Run the ROS deployment policy in a free-base MuJoCo simulation, without ROS."""
import argparse
from contextlib import contextmanager, nullcontext
import time
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supermimic_deploy.ros2_core import JOINTS, SuperMimicCore, quat_inverse_rotate


@contextmanager
def tracking_view(model, data):
    """Create, draw and release the GLFW window on the simulation thread."""
    import glfw

    if not glfw.init():
        raise RuntimeError("GLFW could not open a display")
    window = None
    renderer = None
    try:
        window = glfw.create_window(960, 720, "A00 SuperMimic", None, None)
        if not window:
            raise RuntimeError("Could not create the MuJoCo window")
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        camera = mujoco.MjvCamera()
        camera.distance = 3.0
        camera.elevation = -15
        camera.azimuth = 120
        scene = mujoco.MjvScene(model, maxgeom=max(2000, model.ngeom * 2))
        options = mujoco.MjvOption()
        renderer = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

        def draw():
            glfw.poll_events()
            if glfw.window_should_close(window):
                return False
            camera.lookat[:] = data.qpos[:3]
            mujoco.mjv_updateScene(model, data, options, None, camera,
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)
            width, height = glfw.get_framebuffer_size(window)
            if width and height:
                mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, renderer)
                glfw.swap_buffers(window)
            return True

        yield draw
    finally:
        if renderer is not None:
            renderer.free()
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=ROOT / "assets/A00_robot_mujoco_supermimic.xml")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--viewer", action="store_true", help="Show the simulation at wall-clock speed")
    parser.add_argument("--sim-raw-action", action="store_true",
                        help="Compare raw network actions in simulation; joint targets and actuator forces stay limited")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    core = SuperMimicCore(ROOT, use_raw_policy_action=args.sim_raw_action)
    with np.load(ROOT / "assets/walk2_subject3.npz") as motion:
        names = list(motion["joint_names"].astype(str))
        cols = [names.index(name + "_joint") for name in JOINTS]
        pelvis = list(motion["body_names"].astype(str)).index("pelvis")
        ref_pos = motion["body_pos_w"][:, pelvis]
        ref_height = ref_pos[:, 2]
        ref_joint = motion["joint_pos"][:, cols]
        fps = float(motion["fps"].item())
        data.qpos[:3] = motion["body_pos_w"][0, pelvis]
        data.qpos[3:7] = motion["body_quat_w"][0, pelvis]
        data.qvel[:3] = motion["body_lin_vel_w"][0, pelvis]
        data.qvel[3:6] = quat_inverse_rotate(data.qpos[3:7], motion["body_ang_vel_w"][0, pelvis])
        q0 = motion["joint_pos"][0, cols]
        v0 = motion["joint_vel"][0, cols]

    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected floating root joint at index zero")
    if not np.isclose(fps, 50) or not 0 < args.steps < min(len(ref_height), len(core.goals)):
        raise ValueError("Expected a 50 Hz reference and a step count within the clip")
    substeps = round(0.02 / model.opt.timestep)
    if not np.isclose(substeps * model.opt.timestep, 0.02):
        raise ValueError("Physics timestep must divide the 20 ms policy period")
    if not np.any((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) & (model.geom_bodyid == 0)):
        raise ValueError("Scene needs a world ground plane")
    ids = np.array([model.actuator(n + "_servo").id for n in JOINTS])
    joints = model.actuator_trnid[ids, 0]
    if len(set(joints.tolist())) != 29:
        raise ValueError("Expected 29 distinct driven joints")
    for name, jid, aid in zip(JOINTS, joints, ids):
        if model.joint(int(jid)).name not in (name, name + "_joint"):
            raise ValueError(f"Wrong transmission for {name}")
        if "ankle" in name and not (
            model.actuator_forcelimited[aid]
            and np.allclose(model.actuator_forcerange[aid], [-45, 45])
            and np.isclose(model.actuator_gear[aid, 0], 1)
        ):
            raise ValueError(f"Wrong ankle torque limit for {name}")
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    data.qpos[qa], data.qvel[va] = q0, v0
    mujoco.mj_forward(model, data)

    def sensors():
        gravity = quat_inverse_rotate(data.qpos[3:7], np.array([0., 0., -1.]))
        return data.qpos[qa], data.qvel[va], data.qvel[3:6], gravity

    core.reset(*sensors())
    print(f"scene={args.xml.name} dt={model.opt.timestep} substeps={substeps} default_max={abs(core.default).max():.4f}", flush=True)
    context = tracking_view(model, data) if args.viewer else nullcontext(None)
    with context as draw:
        start = time.monotonic()
        for step in range(1, args.steps + 1):
            action, target, _ = core.step(*sensors())
            data.ctrl[ids] = target
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            if draw is not None:
                if not draw():
                    print("Viewer closed")
                    return
                time.sleep(max(0, start + step * 0.02 - time.monotonic()))
            z = data.qpos[2]
            if not np.isclose(data.time, step * 0.02) or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise SystemExit(f"FAIL: numerical instability at step {step}")
            if step == 1 or step % 25 == 0 or z < 0.3:
                print(f"step={step} t={data.time:.2f} pelvis_z={z:.3f} ref_z={ref_height[step]:.3f} xy_error={np.linalg.norm(data.qpos[:2]-ref_pos[step,:2]):.3f} joint_rms={np.sqrt(np.mean((data.qpos[qa]-ref_joint[step])**2)):.3f} action_norm={np.linalg.norm(action):.3f}", flush=True)
            if z < 0.3 or z > 3.0:
                raise SystemExit(f"FAIL: pelvis height outside diagnostic bounds at step {step}")
    print(f"Completed {args.steps} steps within height bounds")


if __name__ == "__main__":
    main()
