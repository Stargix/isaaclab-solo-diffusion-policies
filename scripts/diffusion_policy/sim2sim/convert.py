"""Convert audited SoloFlat primitives and native inertias to an explicit MJCF.

Reject unsupported/sheared geometry instead of approximating it. No task outcome
is used in conversion. USD joint frames define the zero-coordinate convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .dependencies import load_mujoco

JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                    for joint in ("hip", "thigh", "calf"))
OFFSET = np.array([0, .4, -.8, 0, .4, -.8, 0, -.4, .8, 0, -.4, .8])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(value).reshape(-1))


def quat(matrix):
    return Rotation.from_matrix(matrix).as_quat(scalar_first=True)


def transform(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    result[:3, 3] = position
    return result


def decompose(matrix):
    matrix = np.asarray(matrix, dtype=float)
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale <= 0):
        raise ValueError("Zero transform scale")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ValueError("Sheared/reflected USD transform cannot be silently approximated")
    return matrix[:3, 3], rotation, scale


def build_xml(audit):
    if audit["meters_per_unit"] != 1 or audit["up_axis"] != "Z":
        raise ValueError("Expected metre/Z-up asset")
    if audit["unresolved_inertial_properties"]:
        raise ValueError("Unresolved mass/inertia properties")
    bodies = {b["path"]: b for b in audit["rigid_bodies"]}
    inertias = {q["body"]: q for q in audit["resolved_physx_queries"] if q["valid"]}
    if set(inertias) != set(bodies):
        raise ValueError("Every body must have a valid native inertia query")
    children = {}
    child_paths = set()
    for joint in audit["joints"]:
        p = joint["relationships"]["physics:body0"]
        c = joint["relationships"]["physics:body1"]
        if len(p) != 1 or len(c) != 1 or p[0] not in bodies or c[0] not in bodies:
            raise ValueError("Unsupported world/dangling joint")
        if c[0] in child_paths:
            raise ValueError("Joint graph is not a tree")
        child_paths.add(c[0]); children.setdefault(p[0], []).append((c[0], joint))
    roots = set(bodies) - child_paths
    if len(roots) != 1 or next(iter(roots)).split("/")[-1] != "base":
        raise ValueError("Expected one free base")
    hinge_names = {j["path"].split("/")[-1] for j in audit["joints"]
                   if j["type"] == "PhysicsRevoluteJoint"}
    if hinge_names != set(JOINT_NAMES):
        raise ValueError("Unexpected joint names")
    xml = ET.Element("mujoco", model="solo12_native_usd")
    ET.SubElement(xml, "compiler", angle="radian", inertiafromgeom="false", fusestatic="false")
    ET.SubElement(xml, "option", timestep="0.005", gravity="0 0 -9.81",
                  integrator="implicitfast", solver="Newton", iterations="50", tolerance="1e-10")
    defaults = ET.SubElement(xml, "default")
    ET.SubElement(defaults, "joint", damping="0", frictionloss="0", armature="0.00036207")
    ET.SubElement(defaults, "geom", contype="2", conaffinity="1", friction="1 0.005 0.0001",
                  condim="3", solref="0.02 1", solimp="0.9 0.95 0.001", rgba="0.4 0.5 0.6 1")
    world = ET.SubElement(xml, "worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 0.1",
                  contype="1", conaffinity="2", rgba="0.9 0.9 0.9 1")
    nodes = {}
    visited = set()

    def add(path, parent, relative, joint=None):
        if path in visited:
            raise ValueError("Cyclic joint graph")
        visited.add(path)
        name = path.split("/")[-1]
        pos, rotation, scale = decompose(relative)
        if not np.allclose(scale, 1, atol=1e-6):
            raise ValueError("Scaled body frame needs an explicit separate conversion")
        node = ET.SubElement(parent, "body", name=name, pos=vector(pos), quat=vector(quat(rotation)))
        nodes[path] = node
        if joint is None:
            ET.SubElement(node, "freejoint", name="root")
        elif joint["type"] == "PhysicsRevoluteJoint":
            attr = joint["attributes"]
            local1 = Rotation.from_quat(attr["physics:localRot1"]["value"], scalar_first=True)
            axis = np.eye(3)["XYZ".index(attr["physics:axis"]["value"])]
            limits = np.deg2rad([attr["physics:lowerLimit"]["value"], attr["physics:upperLimit"]["value"]])
            if not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError("Unsupported joint limits")
            ET.SubElement(node, "joint", name=joint["path"].split("/")[-1], type="hinge",
                          pos=vector(attr["physics:localPos1"]["value"]),
                          axis=vector(local1.apply(axis)), range=vector(limits), limited="true")
        elif joint["type"] != "PhysicsFixedJoint":
            raise ValueError(f"Unsupported joint: {joint['type']}")
        inertia = inertias[path]
        inertia_quat = np.asarray(inertia["principal_axes_xyzw"])[[3, 0, 1, 2]]
        ET.SubElement(node, "inertial", pos=vector(inertia["center_of_mass"]),
                      quat=vector(inertia_quat), mass=str(inertia["mass"]),
                      diaginertia=vector(inertia["diagonal_inertia"]))
        for child, child_joint in children.get(path, []):
            attrs = child_joint["attributes"]
            t0 = transform(attrs["physics:localPos0"]["value"], attrs["physics:localRot0"]["value"])
            t1 = transform(attrs["physics:localPos1"]["value"], attrs["physics:localRot1"]["value"])
            add(child, node, t0 @ np.linalg.inv(t1), child_joint)

    root_pose = np.eye(4); root_pose[2, 3] = .35
    add(next(iter(roots)), world, root_pose)
    if visited != set(bodies):
        raise ValueError("Disconnected bodies")
    for collision in audit["collisions"]:
        body_path = collision["body_path"]
        body_world = np.asarray(bodies[body_path]["world_matrix_row_vectors"]).T
        geom_world = np.asarray(collision["world_matrix_row_vectors"]).T
        pos, rotation, scale = decompose(np.linalg.inv(body_world) @ geom_world)
        shape = collision["shape"]; kind = collision["type"]
        if kind == "Cube":
            kind = "box"; size = scale * float(shape["size"]) / 2
        elif kind == "Sphere":
            if not np.allclose(scale, scale[0], atol=1e-7):
                raise ValueError("Ellipsoidal sphere requires explicit mesh conversion")
            kind = "sphere"; size = [float(shape["radius"]) * scale[0]]
        elif kind == "Cylinder":
            axis_index = "XYZ".index(shape.get("axis", "Z"))
            transverse = [i for i in range(3) if i != axis_index]
            if not np.allclose(scale[transverse], scale[transverse[0]], atol=1e-7):
                raise ValueError("Elliptical cylinder requires explicit mesh conversion")
            size = [float(shape["radius"]) * scale[transverse[0]],
                    float(shape["height"]) * scale[axis_index] / 2]
            alignment = (Rotation.from_rotvec([0, np.pi/2, 0]).as_matrix() if axis_index == 0
                         else Rotation.from_rotvec([-np.pi/2, 0, 0]).as_matrix() if axis_index == 1
                         else np.eye(3))
            rotation = rotation @ alignment; kind = "cylinder"
        else:
            raise ValueError(f"Unsupported collision primitive {kind}")
        ET.SubElement(nodes[body_path], "geom", name=name_from_path(collision["path"]),
                      type=kind, pos=vector(pos), quat=vector(quat(rotation)), size=vector(size))
    actuator = ET.SubElement(xml, "actuator")
    for name in JOINT_NAMES:
        ET.SubElement(actuator, "motor", name=name + "_motor", joint=name, gear="1",
                      ctrllimited="true", ctrlrange="-2.65 2.65")
    return xml


def name_from_path(path):
    return path.removeprefix("/Solo_description/").replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dependency-dir")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    xml = build_xml(audit)
    mujoco = load_mujoco(args.dependency_dir)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    if model.nq != 19 or model.nv != 18 or model.nu != 12:
        raise ValueError(f"Unexpected model dimensions: {model.nq}, {model.nv}, {model.nu}")
    expected_mass = sum(q["mass"] for q in audit["resolved_physx_queries"])
    if abs(model.body_mass.sum() - expected_mass) > 1e-7:
        raise ValueError("Conversion changed native total mass")
    output.mkdir(parents=True, exist_ok=True)
    ET.indent(xml)
    model_path = output / "solo12.xml"
    ET.ElementTree(xml).write(model_path, encoding="utf-8", xml_declaration=True)
    manifest = {"schema": 1, "source_sha256": audit["source_sha256"],
                "audit_sha256": sha(args.audit), "mjcf_sha256": sha(model_path),
                "mujoco_version": mujoco.__version__, "mass_kg": float(model.body_mass.sum()),
                "joint_names": JOINT_NAMES, "q_offset": OFFSET.tolist(),
                "physics_dt_s": .005, "policy_dt_s": .020, "kp": 9., "kd": .2,
                "torque_limit_nm": 2.65, "self_collision": False,
                "fk_validated": False, "physical_transfer_validated": False,
                "differences": ["MuJoCo explicit PD versus PhysX implicit actuator",
                                "Different engine contact solvers; friction/contact parameters are not exact equivalents",
                                "USD cylinders imported as native MuJoCo cylinders, not PhysX cooked convex hulls",
                                "PhysX joint velocity limit is not implemented as a state clamp"]}
    (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
