"""Verify converted model FK against three real native PhysX configurations."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .convert import JOINT_NAMES, sha
from .dependencies import load_mujoco


def verify(model_path, native_path, dependency_dir):
    mj = load_mujoco(dependency_dir)
    model = mj.MjModel.from_xml_path(str(Path(model_path).resolve()))
    data = mj.MjData(model)
    native = json.loads(Path(native_path).read_text(encoding="utf-8"))
    if tuple(native["joint_names"]) != JOINT_NAMES:
        raise ValueError("Native joint order mismatch")
    joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    if min(joint_ids) < 0:
        raise ValueError("Missing joint")
    addresses = model.jnt_qposadr[joint_ids]
    cases = []
    for case in native["cases"]:
        mj.mj_resetData(model, data)
        root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "root")
        a = model.jnt_qposadr[root]
        data.qpos[a:a+3] = case["base_pos"]
        data.qpos[a+3:a+7] = case["base_quat_wxyz"]
        data.qpos[addresses] = case["q_policy"]
        mj.mj_forward(model, data)
        positions, angles = [], []
        for name, position, quaternion in zip(native["body_names"], case["body_pos"], case["body_quat_wxyz"]):
            b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                raise ValueError(f"Missing native body {name}")
            positions.append(float(np.linalg.norm(data.xpos[b] - position)))
            actual = Rotation.from_quat(data.xquat[b], scalar_first=True)
            expected = Rotation.from_quat(quaternion, scalar_first=True)
            angles.append(float((expected.inv() * actual).magnitude()))
        cases.append({"name": case["name"], "max_position_error_m": max(positions),
                      "max_orientation_error_rad": max(angles)})
    mass_error = abs(float(model.body_mass.sum()) - sum(native["masses"]))
    body_mass_errors, inertia_eigenvalue_errors = [], []
    for name, mass, inertia in zip(native["body_names"], native["masses"], native["inertias"]):
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        body_mass_errors.append(abs(float(model.body_mass[body_id]) - mass))
        # Native tensor-view inertia uses its inertial frame. Compare eigenvalues
        # without assuming those frames have identical axis ordering.
        expected_eigenvalues = np.linalg.eigvalsh(np.asarray(inertia).reshape(3, 3))
        inertia_eigenvalue_errors.append(float(np.max(np.abs(
            np.sort(model.body_inertia[body_id]) - expected_eigenvalues))))
    result = {"mjcf_sha256": sha(model_path), "native_fk_sha256": sha(native_path),
              "cases": cases, "mass_error_kg": mass_error,
              "position_tolerance_m": .001, "orientation_tolerance_rad": .001,
              "mass_tolerance_kg": 1e-5,
              "max_body_mass_error_kg": max(body_mass_errors),
              "max_inertia_eigenvalue_error_kg_m2": max(inertia_eigenvalue_errors),
              "pass": all(c["max_position_error_m"] <= .001 and c["max_orientation_error_rad"] <= .001
                          for c in cases) and mass_error < 1e-5 and max(body_mass_errors) < 1e-6
                      and max(inertia_eigenvalue_errors) < 1e-8,
              "scope": "Static frames/joint conventions and total mass, not dynamic or contact equivalence"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-fk", required=True)
    parser.add_argument("--dependency-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = verify(args.model, args.native_fk, args.dependency_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
