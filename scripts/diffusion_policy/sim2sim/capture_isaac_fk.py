"""Capture native PhysX kinematics/masses, without policy inference or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    app = AppLauncher(args, fast_shutdown=False).app
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab_tasks.direct.solo12.solo12_env_cfg import SOLO12_CFG, SAFE_INITIAL_JOINT_POS, JOINT_NAMES

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
        cfg = SOLO12_CFG.replace(prim_path="/World/Robot")
        # Kinematics only: suspend in free space, no ground, gravity disabled.
        cfg.spawn.rigid_props.disable_gravity = True
        cfg.init_state.pos = (0., 0., 1.)
        cfg.init_state.joint_pos = SAFE_INITIAL_JOINT_POS
        robot = Articulation(cfg)
        sim.reset()
        robot.update(0.005)
        ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
        if list(names) != list(JOINT_NAMES):
            raise ValueError("Unexpected native joint order")
        safe = torch.tensor([SAFE_INITIAL_JOINT_POS[n] for n in names], device=args.device)
        perturb = torch.linspace(-0.12, 0.12, 12, device=args.device)
        cases = []
        for label, q in (("safe", safe), ("perturb_positive", safe + perturb),
                         ("perturb_negative", safe - perturb)):
            root = robot.data.default_root_state.clone()
            root[:, :3] = torch.tensor([0., 0., 1.], device=args.device)
            root[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=args.device)
            root[:, 7:] = 0
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(q[None], torch.zeros_like(q)[None], joint_ids=ids)
            robot.set_joint_position_target(q[None], joint_ids=ids)
            robot.write_data_to_sim()
            sim.forward()
            robot.update(0.005)
            observed = robot.data.joint_pos[0, ids]
            if not torch.allclose(observed, q, atol=1e-6, rtol=0):
                raise RuntimeError("Native joint write did not preserve FK configuration")
            cases.append({"name": label, "q_policy": observed.cpu().tolist(),
                          "base_pos": robot.data.root_pos_w[0].cpu().tolist(),
                          "base_quat_wxyz": robot.data.root_quat_w[0].cpu().tolist(),
                          "body_pos": robot.data.body_link_pos_w[0].cpu().tolist(),
                          "body_quat_wxyz": robot.data.body_link_quat_w[0].cpu().tolist()})
        output.mkdir(parents=True, exist_ok=True)
        result = {"scope": "Native static FK in gravity-free space; not locomotion evaluation",
                  "joint_names": list(names), "body_names": robot.body_names,
                  "masses": robot.root_physx_view.get_masses()[0].cpu().tolist(),
                  "inertias": robot.root_physx_view.get_inertias()[0].cpu().tolist(),
                  "cases": cases}
        (output / "isaac_fk.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Captured {len(cases)} native FK configurations", flush=True)
    except Exception:
        import traceback
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        # Release tensor views/callbacks before Kit shutdown; otherwise a native
        # PhysX context can remain alive after the capture has finished.
        if "robot" in locals():
            del robot
        if "sim" in locals():
            sim.clear_all_callbacks()
            sim.clear_instance()
        import gc
        gc.collect()
        app.close()


if __name__ == "__main__":
    main()
