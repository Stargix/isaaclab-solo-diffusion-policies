"""Named-joint MuJoCo adapter with explicit per-substep PD and copied state."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .convert import JOINT_NAMES, OFFSET, sha
from .dependencies import load_mujoco


@dataclass(frozen=True)
class State:
    q: np.ndarray
    dq: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    linear_velocity_world: np.ndarray
    angular_velocity_body: np.ndarray
    gravity_body: np.ndarray
    time_s: float

    def proprio(self):
        return np.concatenate((self.q, self.dq, self.angular_velocity_body, self.gravity_body)).astype(np.float32)


@dataclass(frozen=True)
class TickDiagnostics:
    torque_max_nm: float
    saturated_fraction: float
    base_force_max_n: float
    foot_force_max_n: np.ndarray
    failure: str | None


class MujocoBackend:
    def __init__(self, model_path, dependency_dir=None, *, validation_path=None):
        self.mj = load_mujoco(dependency_dir)
        self.model = self.mj.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = self.mj.MjData(self.model)
        if validation_path is not None:
            validation = json.loads(Path(validation_path).read_text())
            if not validation["pass"] or validation["mjcf_sha256"] != sha(model_path):
                raise ValueError("Model has not passed the matching native FK gate")
        if (self.model.nq, self.model.nv, self.model.nu) != (19, 18, 12):
            raise ValueError("Expected one free base and twelve hinges/torque actuators")
        self.joints = self._ids(self.mj.mjtObj.mjOBJ_JOINT, JOINT_NAMES)
        self.actuators = self._ids(self.mj.mjtObj.mjOBJ_ACTUATOR, [n + "_motor" for n in JOINT_NAMES])
        self.q_ids = self.model.jnt_qposadr[self.joints].copy()
        self.dq_ids = self.model.jnt_dofadr[self.joints].copy()
        for joint, actuator in zip(self.joints, self.actuators):
            if self.model.actuator_trnid[actuator, 0] != joint:
                raise ValueError("Actuator transmission mapping differs from named joint")
            if not np.allclose(self.model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]):
                raise ValueError("Unexpected gear: torque must not be scaled twice")
        if not np.allclose(self.model.dof_damping[self.dq_ids], 0):
            raise ValueError("Passive joint damping would duplicate Kd")
        if not np.allclose(self.model.dof_armature[self.dq_ids], .00036207):
            raise ValueError("Unexpected joint armature")
        if abs(self.model.opt.timestep - .005) > 1e-12:
            raise ValueError("Pinned physics timestep is .005 s")
        self.base = self._ids(self.mj.mjtObj.mjOBJ_BODY, ["base"])[0]
        self.feet = self._ids(self.mj.mjtObj.mjOBJ_BODY, [leg + "_calf" for leg in ("FL", "FR", "RL", "RR")])
        root = self._ids(self.mj.mjtObj.mjOBJ_JOINT, ["root"])[0]
        self.root_q = int(self.model.jnt_qposadr[root])
        self.root_dq = int(self.model.jnt_dofadr[root])
        self.reset()

    def _ids(self, kind, names):
        ids = np.array([self.mj.mj_name2id(self.model, kind, name) for name in names])
        if np.any(ids < 0) or len(set(ids)) != len(ids):
            raise ValueError(f"Missing/duplicate mapping: {names}")
        return ids

    def reset(self, *, position=(0., 0., .35), quaternion=(1., 0., 0., 0.), q=None, dq=None):
        self.mj.mj_resetData(self.model, self.data)
        quaternion = np.asarray(quaternion, dtype=float)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1):
            raise ValueError("Reset needs a unit wxyz quaternion")
        self.data.qpos[self.root_q:self.root_q+3] = position
        self.data.qpos[self.root_q+3:self.root_q+7] = quaternion
        self.data.qpos[self.q_ids] = OFFSET if q is None else np.asarray(q)
        if dq is not None:
            self.data.qvel[self.dq_ids] = dq
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.mj.mj_forward(self.model, self.data)
        return self.read_state()

    def read_state(self):
        local = np.zeros(6); world = np.zeros(6)
        # XBODY is the regular link frame, not the centre-of-mass/inertial frame.
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_XBODY, int(self.base), local, 1)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_XBODY, int(self.base), world, 0)
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return State(self.data.qpos[self.q_ids].copy(), self.data.qvel[self.dq_ids].copy(),
                     self.data.xpos[self.base].copy(), self.data.xquat[self.base].copy(),
                     world[3:].copy(), local[:3].copy(), rotation.T @ np.array([0., 0., -1.]),
                     float(self.data.time))

    def contact_forces(self):
        forces = np.zeros((self.model.nbody, 3))
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            force = np.zeros(6)
            self.mj.mj_contactForce(self.model, self.data, index, force)
            world_force = contact.frame.reshape(3, 3).T @ force[:3]
            b0, b1 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            forces[b0] -= world_force; forces[b1] += world_force
        return np.linalg.norm(forces, axis=1)

    def step_policy_tick(self, target, *, substeps=4):
        if substeps not in (1, 4):
            raise ValueError("Only one physics step or one four-step policy tick is supported")
        target = np.asarray(target, dtype=float)
        if target.shape != (12,) or not np.isfinite(target).all():
            raise ValueError("Expected finite 12-joint target")
        torque_max = base_max = 0.; foot_max = np.zeros(4); saturated = count = 0
        failure = None
        for _ in range(substeps):
            raw_torque = 9 * (target - self.data.qpos[self.q_ids]) - .2 * self.data.qvel[self.dq_ids]
            torque = np.clip(raw_torque, -2.65, 2.65)
            self.data.ctrl[self.actuators] = torque
            saturated += int(np.count_nonzero(np.abs(raw_torque) >= 2.65)); count += 12
            torque_max = max(torque_max, float(np.max(np.abs(torque))))
            previous_time = float(self.data.time)
            self.mj.mj_step(self.model, self.data)
            force = self.contact_forces()
            base_max = max(base_max, float(force[self.base])); foot_max = np.maximum(foot_max, force[self.feet])
            if abs(self.data.time - previous_time - .005) > 1e-9 or not (
                np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
            ):
                failure = "numerical_reset_or_nonfinite"
            self.mj.mj_forward(self.model, self.data)
            state = self.read_state()
            tilt = np.rad2deg(np.arccos(np.clip(-state.gravity_body[2], -1, 1)))
            if failure is None and not (.08 <= state.position[2] <= .50 and
                                       np.linalg.norm(state.linear_velocity_world[:2]) <= 5 and tilt <= 60):
                failure = "physical_sanity"
            if failure is None and base_max > 1.:
                failure = "base_contact"
            if failure is not None:
                break
        return self.read_state(), TickDiagnostics(torque_max, saturated / count, base_max, foot_max.copy(), failure)

    def close(self):
        # No viewer or process is opened by the backend.
        pass
