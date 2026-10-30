from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from motion_edit.contact.schema import ContactSurfaceRecord


def parse_box_descriptor(descriptor: str) -> tuple[str, list[float], list[float]]:
    try:
        object_id, center_text, size_text = descriptor.split(":")
    except ValueError as exc:
        raise ValueError("box descriptor must be object_id:cx,cy,cz:sx,sy,sz") from exc
    center = [float(item) for item in center_text.split(",")]
    size = [float(item) for item in size_text.split(",")]
    if len(center) != 3 or len(size) != 3:
        raise ValueError("box descriptor center and size must have three components")
    if any(item <= 0.0 for item in size):
        raise ValueError("box size components must be positive")
    return object_id, center, size


def box_surfaces(
    *,
    motion_id: str,
    object_id: str,
    center: list[float],
    size: list[float],
    include_sides: bool = True,
    source: str = "box_surface_catalog",
) -> list[ContactSurfaceRecord]:
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    surfaces = [
        ContactSurfaceRecord(
            motion_id=motion_id,
            surface_id=f"{object_id}_top",
            object_id=object_id,
            surface_type="box_face",
            origin=[cx, cy, cz + hz],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-hx, hx], "v": [-hy, hy]},
            source=source,
        )
    ]
    if include_sides:
        surfaces.extend(
            [
                ContactSurfaceRecord(
                    motion_id=motion_id,
                    surface_id=f"{object_id}_pos_x",
                    object_id=object_id,
                    surface_type="box_face",
                    origin=[cx + hx, cy, cz],
                    normal=[1.0, 0.0, 0.0],
                    tangent_u=[0.0, 1.0, 0.0],
                    tangent_v=[0.0, 0.0, 1.0],
                    bounds={"u": [-hy, hy], "v": [-hz, hz]},
                    source=source,
                ),
                ContactSurfaceRecord(
                    motion_id=motion_id,
                    surface_id=f"{object_id}_neg_x",
                    object_id=object_id,
                    surface_type="box_face",
                    origin=[cx - hx, cy, cz],
                    normal=[-1.0, 0.0, 0.0],
                    tangent_u=[0.0, 1.0, 0.0],
                    tangent_v=[0.0, 0.0, 1.0],
                    bounds={"u": [-hy, hy], "v": [-hz, hz]},
                    source=source,
                ),
                ContactSurfaceRecord(
                    motion_id=motion_id,
                    surface_id=f"{object_id}_pos_y",
                    object_id=object_id,
                    surface_type="box_face",
                    origin=[cx, cy + hy, cz],
                    normal=[0.0, 1.0, 0.0],
                    tangent_u=[1.0, 0.0, 0.0],
                    tangent_v=[0.0, 0.0, 1.0],
                    bounds={"u": [-hx, hx], "v": [-hz, hz]},
                    source=source,
                ),
                ContactSurfaceRecord(
                    motion_id=motion_id,
                    surface_id=f"{object_id}_neg_y",
                    object_id=object_id,
                    surface_type="box_face",
                    origin=[cx, cy - hy, cz],
                    normal=[0.0, -1.0, 0.0],
                    tangent_u=[1.0, 0.0, 0.0],
                    tangent_v=[0.0, 0.0, 1.0],
                    bounds={"u": [-hx, hx], "v": [-hz, hz]},
                    source=source,
                ),
            ]
        )
    return surfaces


def _parse_floats(text: str | None, *, default: tuple[float, ...]) -> list[float]:
    if not text:
        return [float(item) for item in default]
    values = [float(item) for item in text.split()]
    return values


def _matmul3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def _sub3(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(3)]


def _add3(a: list[float], b: list[float]) -> list[float]:
    return [a[index] + b[index] for index in range(3)]


def _scale3(a: list[float], scale: float) -> list[float]:
    return [a[index] * scale for index in range(3)]


def _dot3(a: list[float], b: list[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _cross3(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize3(vector: list[float]) -> list[float]:
    norm = math.sqrt(_dot3(vector, vector))
    if norm == 0.0:
        raise ValueError("zero-length vector cannot be normalized")
    return [item / norm for item in vector]


def _rotation_from_rpy(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _load_obj_vertices(path: Path) -> list[list[float]]:
    vertices: list[list[float]] = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"{path} contains no OBJ vertices")
    return vertices


def _load_obj_mesh(path: Path) -> tuple[list[list[float]], list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                face = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                if len(face) >= 3:
                    faces.append(face)
    if not vertices:
        raise ValueError(f"{path} contains no OBJ vertices")
    if not faces:
        raise ValueError(f"{path} contains no OBJ faces")
    return vertices, faces


def _transform_vertices(
    vertices: list[list[float]],
    *,
    scale: list[float],
    xyz: list[float],
    rpy: list[float],
) -> list[list[float]]:
    rotation = _rotation_from_rpy(rpy)
    transformed = []
    for vertex in vertices:
        scaled = [vertex[index] * scale[index] for index in range(3)]
        rotated = _matmul3(rotation, scaled)
        transformed.append([rotated[index] + xyz[index] for index in range(3)])
    return transformed


def _face_normal(vertices: list[list[float]], face: list[int]) -> list[float]:
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return _normalize3(_cross3(_sub3(b, a), _sub3(c, a)))


def _surface_name_from_normal(normal: list[float], fallback: int) -> str:
    if normal[2] > 0.9:
        return "top"
    if normal[2] < -0.9:
        return "bottom"
    return f"side_{fallback:02d}"


def _surface_from_face_group(
    *,
    motion_id: str,
    object_id: str,
    vertices: list[list[float]],
    face_indices: list[int],
    faces: list[list[int]],
    normal: list[float],
    suffix: str,
    source: str,
    metadata: dict,
) -> ContactSurfaceRecord:
    group_vertex_ids = sorted({vertex_id for face_index in face_indices for vertex_id in faces[face_index]})
    group_vertices = [vertices[index] for index in group_vertex_ids]
    origin0 = group_vertices[0]
    tangent_u = None
    for face_index in face_indices:
        face = faces[face_index]
        for a, b in zip(face, face[1:] + face[:1]):
            edge = _sub3(vertices[b], vertices[a])
            edge_in_plane = _sub3(edge, _scale3(normal, _dot3(edge, normal)))
            if _dot3(edge_in_plane, edge_in_plane) > 1e-12:
                tangent_u = _normalize3(edge_in_plane)
                break
        if tangent_u is not None:
            break
    if tangent_u is None:
        tangent_u = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
    tangent_v = _normalize3(_cross3(normal, tangent_u))
    coords = [(_dot3(_sub3(vertex, origin0), tangent_u), _dot3(_sub3(vertex, origin0), tangent_v)) for vertex in group_vertices]
    min_u = min(coord[0] for coord in coords)
    max_u = max(coord[0] for coord in coords)
    min_v = min(coord[1] for coord in coords)
    max_v = max(coord[1] for coord in coords)
    center_u = (min_u + max_u) / 2.0
    center_v = (min_v + max_v) / 2.0
    origin = _add3(origin0, _add3(_scale3(tangent_u, center_u), _scale3(tangent_v, center_v)))
    polygon = sorted(
        (
            {
                "world": vertex,
                "u": _dot3(_sub3(vertex, origin), tangent_u),
                "v": _dot3(_sub3(vertex, origin), tangent_v),
            }
            for vertex in group_vertices
        ),
        key=lambda item: math.atan2(float(item["v"]), float(item["u"])),
    )
    return ContactSurfaceRecord(
        motion_id=motion_id,
        surface_id=f"{object_id}_{suffix}",
        object_id=object_id,
        surface_type="mesh_face",
        origin=origin,
        normal=normal,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        bounds={"u": [min_u - center_u, max_u - center_u], "v": [min_v - center_v, max_v - center_v]},
        source=source,
        metadata={
            **metadata,
            "face_indices": face_indices,
            "surface_extraction": "obj_face_groups",
            "polygon_world": [item["world"] for item in polygon],
            "polygon_surface_coordinates": [{"u": item["u"], "v": item["v"]} for item in polygon],
        },
    )


def _mesh_face_surfaces(
    *,
    motion_id: str,
    object_id: str,
    vertices: list[list[float]],
    faces: list[list[int]],
    include_sides: bool,
    include_downward: bool,
    source: str,
    metadata: dict,
) -> list[ContactSurfaceRecord]:
    groups: dict[tuple[float, float, float, float], list[int]] = defaultdict(list)
    normals: dict[tuple[float, float, float, float], list[float]] = {}
    for index, face in enumerate(faces):
        normal = _face_normal(vertices, face)
        if not include_downward and normal[2] < -0.5:
            continue
        if not include_sides and normal[2] < 0.5:
            continue
        plane_offset = _dot3(normal, vertices[face[0]])
        key = (round(normal[0], 6), round(normal[1], 6), round(normal[2], 6), round(plane_offset, 6))
        groups[key].append(index)
        normals[key] = normal
    surfaces: list[ContactSurfaceRecord] = []
    side_index = 0
    for key, face_indices in sorted(groups.items()):
        normal = normals[key]
        suffix = _surface_name_from_normal(normal, side_index)
        if suffix.startswith("side_"):
            side_index += 1
        surfaces.append(
            _surface_from_face_group(
                motion_id=motion_id,
                object_id=object_id,
                vertices=vertices,
                faces=faces,
                face_indices=face_indices,
                normal=normal,
                suffix=suffix,
                source=source,
                metadata=metadata,
            )
        )
    return surfaces


def ground_surface(
    *,
    motion_id: str,
    half_extent: float = 10.0,
    z: float = 0.0,
    source: str = "ground_plane_surface_catalog",
) -> ContactSurfaceRecord:
    return ContactSurfaceRecord(
        motion_id=motion_id,
        surface_id="terrain_ground_z0",
        object_id="terrain_ground",
        surface_type="plane",
        origin=[0.0, 0.0, float(z)],
        normal=[0.0, 0.0, 1.0],
        tangent_u=[1.0, 0.0, 0.0],
        tangent_v=[0.0, 1.0, 0.0],
        bounds={"u": [-float(half_extent), float(half_extent)], "v": [-float(half_extent), float(half_extent)]},
        source=source,
        metadata={"surface_extraction": "explicit_ground_plane", "ground_z": float(z), "half_extent": float(half_extent)},
    )


def surfaces_from_urdf_meshes(
    *,
    motion_id: str,
    urdf_path: str | Path,
    include_sides: bool = True,
    include_downward: bool = False,
    include_ground: bool = True,
    ground_z: float = 0.0,
    ground_half_extent: float = 10.0,
    source: str = "urdf_mesh_surface_catalog",
) -> list[ContactSurfaceRecord]:
    urdf = Path(urdf_path).expanduser().resolve()
    root = ET.parse(urdf).getroot()
    surfaces: list[ContactSurfaceRecord] = []
    mesh_index = 0
    for link in root.findall("link"):
        link_name = link.attrib.get("name") or f"link_{mesh_index}"
        geom_parents = link.findall("collision") or link.findall("visual")
        for geom_parent in geom_parents:
            geometry = geom_parent.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh_path = urdf.parent / mesh_path
            if mesh_path.suffix.lower() != ".obj":
                continue
            origin = geom_parent.find("origin")
            xyz = _parse_floats(origin.attrib.get("xyz") if origin is not None else None, default=(0.0, 0.0, 0.0))
            rpy = _parse_floats(origin.attrib.get("rpy") if origin is not None else None, default=(0.0, 0.0, 0.0))
            scale = _parse_floats(mesh.attrib.get("scale"), default=(1.0, 1.0, 1.0))
            if len(xyz) != 3 or len(rpy) != 3 or len(scale) != 3:
                raise ValueError(f"{urdf}: mesh origin/scale must have three components")
            raw_vertices, faces = _load_obj_mesh(mesh_path)
            vertices = _transform_vertices(raw_vertices, scale=scale, xyz=xyz, rpy=rpy)
            object_id = f"{link_name}_{mesh_index}"
            surfaces.extend(
                _mesh_face_surfaces(
                    motion_id=motion_id,
                    object_id=object_id,
                    vertices=vertices,
                    faces=faces,
                    include_sides=include_sides,
                    include_downward=include_downward,
                    source=source,
                    metadata={
                        "urdf_path": str(urdf),
                        "mesh_path": str(mesh_path),
                        "link_name": link_name,
                        "geometry_kind": geom_parent.tag,
                        "mesh_scale": scale,
                        "origin_xyz": xyz,
                        "origin_rpy": rpy,
                    },
                )
            )
            mesh_index += 1
    if include_ground:
        surfaces.append(ground_surface(motion_id=motion_id, half_extent=ground_half_extent, z=ground_z))
    if not surfaces:
        raise ValueError(f"{urdf} has no supported OBJ mesh collision/visual surfaces")
    seen: set[tuple[str, str]] = set()
    unique: list[ContactSurfaceRecord] = []
    for surface in surfaces:
        key = (surface.surface_id, str(surface.metadata.get("mesh_path")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(surface)
    return unique


def surfaces_from_terrain_metadata(*_args, **_kwargs) -> list[ContactSurfaceRecord]:
    raise NotImplementedError("terrain metadata surface loading is not implemented yet")


def surfaces_from_urdf_collision_boxes(*_args, **_kwargs) -> list[ContactSurfaceRecord]:
    raise NotImplementedError("URDF primitive box surface loading is not implemented yet; use surfaces_from_urdf_meshes for OBJ terrain meshes")


def surfaces_from_obj_mesh_faces(
    *,
    motion_id: str,
    obj_path: str | Path,
    include_sides: bool = True,
    include_downward: bool = False,
    include_ground: bool = True,
    ground_z: float = 0.0,
    ground_half_extent: float = 10.0,
    source: str = "obj_mesh_surface_catalog",
) -> list[ContactSurfaceRecord]:
    mesh_path = Path(obj_path).expanduser().resolve()
    vertices, faces = _load_obj_mesh(mesh_path)
    surfaces = _mesh_face_surfaces(
        motion_id=motion_id,
        object_id=mesh_path.stem,
        vertices=vertices,
        faces=faces,
        include_sides=include_sides,
        include_downward=include_downward,
        source=source,
        metadata={"mesh_path": str(mesh_path)},
    )
    if include_ground:
        surfaces.append(ground_surface(motion_id=motion_id, half_extent=ground_half_extent, z=ground_z))
    if not surfaces:
        raise ValueError(f"{mesh_path} has no eligible mesh surfaces")
    return surfaces


def surfaces_from_heightfield_patches(*_args, **_kwargs) -> list[ContactSurfaceRecord]:
    raise NotImplementedError("heightfield surface patch loading is not implemented yet")
