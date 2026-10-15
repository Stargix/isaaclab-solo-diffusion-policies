"""Declared nominal/mass-friction/velocity-impulse tests; no outcome tuning."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Perturbation:
    name: str
    mass_scale: float = 1.
    friction_scale: float = 1.
    lateral_velocity_delta_mps: float = 0.
    impulse_times_s: tuple = (5., 12.)

    def impulse(self, step, dt=.02):
        for i, time in enumerate(self.impulse_times_s):
            if step == round(time/dt):
                return self.lateral_velocity_delta_mps * (1 if i == 0 else -1)
        return 0.


PROTOCOLS = {p.name: p for p in (Perturbation("nominal"),
    Perturbation("mass_friction", mass_scale=1.15, friction_scale=.6),
    Perturbation("lateral_impulse", lateral_velocity_delta_mps=.2))}


def configure_mujoco(backend, name):
    p = PROTOCOLS[name]; b = backend
    before = float(b.model.body_mass.sum())
    if p.mass_scale != 1 or p.friction_scale != 1:
        b.model.body_mass[1:] *= p.mass_scale
        b.model.body_inertia[1:] *= p.mass_scale
        # MuJoCo combines sliding friction by max: scale floor AND robot.
        b.model.geom_friction[:, 0] *= p.friction_scale
        b.mj.mj_setConst(b.model, b.data)
        b.reset()
    return dict(protocol=asdict(p), mass_before_kg=before, mass_after_kg=float(b.model.body_mass.sum()))


def impulse_mujoco(backend, name, step):
    delta = PROTOCOLS[name].impulse(step)
    if not delta:
        return False
    from scipy.spatial.transform import Rotation
    yaw = Rotation.from_quat(backend.read_state().quaternion, scalar_first=True).as_euler("xyz")[2]
    backend.data.qvel[backend.root_dq:backend.root_dq+2] += delta * np.array([-np.sin(yaw), np.cos(yaw)])
    backend.mj.mj_forward(backend.model, backend.data)
    return True


def configure_isaac(raw_env, name):
    import torch
    p = PROTOCOLS[name]; view = raw_env._robot.root_physx_view
    ids = torch.arange(raw_env.num_envs, dtype=torch.int32, device="cpu")
    masses = view.get_masses().clone(); inertias = view.get_inertias().clone()
    materials = view.get_material_properties().clone()
    before = masses.sum(-1).tolist()
    if p.mass_scale != 1:
        view.set_masses(masses*p.mass_scale, ids)
        view.set_inertias(inertias*p.mass_scale, ids)
    if p.friction_scale != 1:
        # Native plane uses multiply with coefficient 1: scale robot only.
        materials[..., :2] *= p.friction_scale
        view.set_material_properties(materials, ids)
    observed = view.get_masses()
    if not torch.allclose(observed, masses*p.mass_scale, atol=1e-6):
        raise RuntimeError("Native mass perturbation did not apply")
    if not torch.allclose(view.get_inertias(), inertias*p.mass_scale, atol=1e-8, rtol=1e-5):
        raise RuntimeError("Native inertia perturbation did not apply")
    actual_material = view.get_material_properties()
    if not torch.allclose(actual_material, materials, atol=1e-6):
        raise RuntimeError("Native friction perturbation did not apply")
    return dict(protocol=asdict(p), mass_before_kg=before, mass_after_kg=observed.sum(-1).tolist(),
                sliding_material_range=[float(actual_material[..., :2].min()), float(actual_material[..., :2].max())],
                inertia_readback_verified=True, friction_readback_verified=True)


def impulse_isaac(raw_env, name, step):
    import torch
    delta = PROTOCOLS[name].impulse(step)
    if not delta:
        return False
    robot = raw_env._robot
    q = robot.data.root_quat_w
    yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
    velocity = robot.data.root_vel_w.clone()
    velocity[:, 0] -= delta*torch.sin(yaw)
    velocity[:, 1] += delta*torch.cos(yaw)
    robot.write_root_velocity_to_sim(velocity)
    return True
