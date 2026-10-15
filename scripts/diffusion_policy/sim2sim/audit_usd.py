"""Inspect the actual Isaac USD before attempting a cross-engine conversion.

Run through isaaclab.bat; this does not step physics or modify the source USD.
Authored USD values are not necessarily resolved PhysX inertial properties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _serial(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "GetReal"):
        return [float(value.GetReal()), *map(float, value.GetImaginary())]
    try:
        return [_serial(item) for item in value]
    except TypeError:
        return str(value)


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-physx", action="store_true",
                        help="Also query resolved PhysX mass properties; never invent missing inertia.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    source = Path(args.usd).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    app = AppLauncher(args, fast_shutdown=False).app
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools

        stage = Usd.Stage.Open(str(source))
        if stage is None:
            raise RuntimeError(f"Cannot compose USD: {source}")
        bodies, joints, collisions = [], [], []
        xforms = UsdGeom.XformCache()
        for prim in stage.Traverse():
            attrs = {
                attr.GetName(): {
                    "value": _serial(attr.Get()),
                    "authored": attr.HasAuthoredValueOpinion(),
                }
                for attr in prim.GetAttributes()
                if attr.GetName().startswith(("physics:", "physx"))
            }
            record = {"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                      "attributes": attrs}
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                record["world_matrix_row_vectors"] = _serial(xforms.GetLocalToWorldTransform(prim))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(record)
            if prim.IsA(UsdPhysics.Joint):
                record["relationships"] = {
                    rel.GetName(): [str(p) for p in rel.GetTargets()]
                    for rel in prim.GetRelationships()
                }
                joints.append(record)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                parent = prim
                while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    parent = parent.GetParent()
                if not parent:
                    raise ValueError(f"Collider has no rigid parent: {prim.GetPath()}")
                record["body_path"] = str(parent.GetPath())
                record["shape"] = {name: _serial(prim.GetAttribute(name).Get())
                                   for name in ("radius", "height", "size", "axis")
                                   if prim.GetAttribute(name)}
                collisions.append(record)
        layers = []
        for layer in stage.GetUsedLayers():
            path = Path(layer.realPath) if layer.realPath else None
            layers.append({"identifier": layer.identifier,
                           "resolved_path": str(path) if path else None,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                           if path and path.is_file() else None})
        default_prim = str(stage.GetDefaultPrim().GetPath())
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        up_axis = str(UsdGeom.GetStageUpAxis(stage))
        resolved = []
        if args.query_physx:
            from omni.physx import get_physx_property_query_interface
            from omni.physx.bindings._physx import PhysxPropertyQueryMode, PhysxPropertyQueryResult

            cache = UsdUtils.StageCache.Get()
            stage_id = cache.Insert(stage).ToLongInt()
            try:
                for body in bodies:
                    item = {"body": body["path"], "valid": False}

                    def callback(info, item=item):
                        item["result"] = str(info.result)
                        if info.result == PhysxPropertyQueryResult.VALID:
                            item.update(valid=True, mass=float(info.mass),
                                        center_of_mass=_serial(info.center_of_mass),
                                        diagonal_inertia=_serial(info.inertia),
                                        principal_axes_xyzw=_serial(info.principal_axes))

                    get_physx_property_query_interface().query_prim(
                        stage_id=stage_id,
                        prim_id=PhysicsSchemaTools.sdfPathToInt(body["path"]),
                        query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS,
                        rigid_body_fn=callback,
                    )
                    # The query can complete on an app update rather than inline.
                    for _ in range(4):
                        if "result" in item:
                            break
                        app.update()
                    resolved.append(item)
            finally:
                cache.Erase(stage)
        unresolved = []
        for body in bodies:
            attrs = body["attributes"]
            mass = attrs.get("physics:mass", {}).get("value")
            inertia = attrs.get("physics:diagonalInertia", {}).get("value")
            if not isinstance(mass, (int, float)) or mass <= 0:
                unresolved.append({"body": body["path"], "property": "mass"})
            if not isinstance(inertia, list) or len(inertia) != 3 or not all(
                isinstance(v, (int, float)) and v > 0 for v in inertia
            ):
                unresolved.append({"body": body["path"], "property": "diagonalInertia"})
        resolved_by_body = {item["body"]: item for item in resolved if item["valid"]}
        remaining = []
        for item in unresolved:
            query = resolved_by_body.get(item["body"], {})
            value = query.get("mass" if item["property"] == "mass" else "diagonal_inertia")
            if item["property"] == "mass":
                valid = isinstance(value, (int, float)) and value > 0
            else:
                valid = isinstance(value, list) and len(value) == 3 and all(
                    isinstance(v, (int, float)) and v > 0 for v in value
                )
            if not valid:
                remaining.append(item)
        result = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "default_prim": default_prim,
            "meters_per_unit": meters_per_unit,
            "up_axis": up_axis,
            "used_layers": layers, "rigid_bodies": bodies, "joints": joints,
            "collisions": collisions,
            "authored_nonpositive_inertial_properties": unresolved,
            "unresolved_inertial_properties": remaining,
            "resolved_physx_queries": resolved,
            "physical_sim2sim_validated": False,
            "limitations": [
                "Read-only USD inspection; optional PhysX queries are recorded separately.",
                "No conversion, cross-engine FK test, contacts or policy rollout performed.",
                "Positive USD inertias alone do not establish model equivalence.",
            ],
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "asset_audit.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps({"bodies": len(bodies), "joints": len(joints),
                          "collisions": len(collisions), "unresolved": len(remaining),
                          "output": str(output)}, indent=2), flush=True)
    except Exception as error:
        import traceback

        output.mkdir(parents=True, exist_ok=True)
        (output / "audit_failure.json").write_text(json.dumps({
            "source": str(source), "physical_sim2sim_validated": False,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        }, indent=2), encoding="utf-8")
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
