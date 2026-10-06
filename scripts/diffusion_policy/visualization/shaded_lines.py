"""Shaded, non-colliding line geometry for the optional clean viewer preset."""
from __future__ import annotations
import numpy as np


def tube_geometry(starts, ends, diameters, *, sides=6):
    """Preserve line centre coordinates, including vertical height changes."""
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    widths = np.asarray(diameters, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape:
        raise ValueError("Line endpoints must have matching (N, 3) shapes.")
    if widths.shape != (len(starts),) or sides < 3:
        raise ValueError("One diameter per segment and at least three sides are required.")
    if not all(np.isfinite(x).all() for x in (starts, ends, widths)) or (widths <= 0).any():
        raise ValueError("Geometry must be finite and diameters positive.")
    delta = ends - starts
    length = np.linalg.norm(delta, axis=1)
    keep = length > 1e-8
    starts, ends, widths, delta, length = starts[keep], ends[keep], widths[keep], delta[keep], length[keep]
    if not len(starts):
        return np.empty((0, 3), dtype=np.float32), []
    direction = delta / length[:, None]
    reference = np.tile([0., 0., 1.], (len(starts), 1))
    reference[np.abs(direction[:, 2]) > .9] = [0., 1., 0.]
    lateral = np.cross(direction, reference)
    lateral /= np.linalg.norm(lateral, axis=1)[:, None]
    normal = np.cross(direction, lateral)
    angles = np.arange(sides) * (2*np.pi/sides)
    radial = (np.cos(angles)[None, :, None]*lateral[:, None, :]
              + np.sin(angles)[None, :, None]*normal[:, None, :]) * widths[:, None, None]/2
    points = np.concatenate([starts[:, None, :]+radial, ends[:, None, :]+radial], axis=1)
    faces = []
    for segment in range(len(starts)):
        offset = segment*2*sides
        for side in range(sides):
            following = (side+1)%sides
            faces.extend([offset+side, offset+following, offset+sides+following, offset+sides+side])
    return points.reshape(-1, 3).astype(np.float32), faces


class ShadedLineDraw:
    """Reusable USD mesh pool matching debug draw's line interface.

    Positions come directly from PathDebugVisualizer. No collision or rigid-body
    APIs are attached; only material, thickness and rendering backend differ.
    """

    def __init__(self):
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        import omni.usd
        self.Gf, self.Sdf, self.UsdGeom, self.UsdShade = Gf, Sdf, UsdGeom, UsdShade
        self.stage = omni.usd.get_context().get_stage()
        self.meshes, self.materials, self.cursor = [], {}, 0

    def clear_lines(self):
        self.cursor = 0
        for mesh in self.meshes:
            mesh.GetVisibilityAttr().Set("invisible")

    def draw_lines(self, starts, ends, colors, widths):
        if not len(starts):
            return
        if not (len(starts) == len(ends) == len(colors) == len(widths)):
            raise ValueError("All line attributes must have the same length.")
        groups = {}
        for index, color in enumerate(colors):
            groups.setdefault(tuple(color[:3]), []).append(index)
        for color, indices in groups.items():
            selected = np.asarray(indices)
            points, faces = tube_geometry(np.asarray(starts)[selected], np.asarray(ends)[selected],
                                          np.asarray(widths)[selected]*.003)
            if not len(points):
                continue
            if self.cursor == len(self.meshes):
                mesh = self.UsdGeom.Mesh.Define(self.stage, f"/World/Visuals/PathDebug/clean_lines_{self.cursor}")
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateDoubleSidedAttr(True)
                self.meshes.append(mesh)
            mesh = self.meshes[self.cursor]
            self.cursor += 1
            mesh.GetPointsAttr().Set([self.Gf.Vec3f(*map(float, p)) for p in points])
            mesh.GetFaceVertexCountsAttr().Set([4]*(len(faces)//4))
            mesh.GetFaceVertexIndicesAttr().Set(faces)
            mesh.GetVisibilityAttr().Set("inherited")
            if color not in self.materials:
                path = f"/World/Visuals/PathDebug/Materials/clean_{len(self.materials)}"
                material = self.UsdShade.Material.Define(self.stage, path)
                shader = self.UsdShade.Shader.Define(self.stage, path+"/Surface")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(self.Gf.Vec3f(*color))
                shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(.7)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                self.materials[color] = material
            self.UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials[color])
