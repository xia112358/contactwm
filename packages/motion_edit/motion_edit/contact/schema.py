from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ContactEventType = Literal[
    "touchdown",
    "liftoff",
    "support_switch",
    "active_change",
    "contact_gain",
    "contact_loss",
]
ContactAnchorRole = Literal["active", "support", "transition", "unknown"]
ContactTransitionType = Literal[
    "step",
    "reach",
    "pull",
    "support_transfer",
    "recovery",
    "micro_adjust",
    "unknown",
]
ContactSurfaceType = Literal["plane", "box_face", "mesh_face", "heightfield", "unknown"]
FRAME_INTERVAL_SEMANTICS = "half_open_[start_frame,end_frame)"


@dataclass(frozen=True)
class ContactSurfaceRecord:
    motion_id: str
    surface_id: str
    object_id: str | None
    surface_type: ContactSurfaceType
    origin: list[float]
    normal: list[float]
    tangent_u: list[float]
    tangent_v: list[float]
    bounds: dict[str, Any] | None = None
    source: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.surface_id:
            raise ValueError("surface_id is required")
        for name in ("origin", "normal", "tangent_u", "tangent_v"):
            value = getattr(self, name)
            if len(value) != 3:
                raise ValueError(f"{self.surface_id}: {name} must have length 3")
        for name in ("normal", "tangent_u", "tangent_v"):
            value = getattr(self, name)
            norm = sum(float(item) * float(item) for item in value) ** 0.5
            if abs(norm - 1.0) > 1e-6:
                raise ValueError(f"{self.surface_id}: {name} must be normalized")
        normal = [float(item) for item in self.normal]
        tangent_u = [float(item) for item in self.tangent_u]
        tangent_v = [float(item) for item in self.tangent_v]
        dot_nu = sum(normal[index] * tangent_u[index] for index in range(3))
        dot_nv = sum(normal[index] * tangent_v[index] for index in range(3))
        if abs(dot_nu) > 1e-6:
            raise ValueError(f"{self.surface_id}: tangent_u must be orthogonal to normal")
        if abs(dot_nv) > 1e-6:
            raise ValueError(f"{self.surface_id}: tangent_v must be orthogonal to normal")
        if self.bounds is not None:
            for axis in ("u", "v"):
                value = self.bounds.get(axis)
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(f"{self.surface_id}: bounds[{axis!r}] must be a two-item list")
                if float(value[1]) < float(value[0]):
                    raise ValueError(f"{self.surface_id}: bounds[{axis!r}] max must be >= min")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ContactEventRecord:
    motion_id: str
    event_id: str
    frame: int
    body: str
    event_type: ContactEventType
    contact_before: list[str] = field(default_factory=list)
    contact_after: list[str] = field(default_factory=list)
    source: str = "contact_mask"
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.frame < 0:
            raise ValueError(f"{self.event_id}: frame must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ContactAnchorRecord:
    """Contact interval using the repository-wide half-open ``[start, end)`` convention."""

    motion_id: str
    anchor_id: str
    body: str
    start_frame: int
    end_frame: int
    role: ContactAnchorRole = "unknown"
    world_position: list[float] | None = None
    object_position: list[float] | None = None
    object_id: str | None = None
    normal: list[float] | None = None
    surface_id: str | None = None
    surface_type: ContactSurfaceType = "unknown"
    surface_normal: list[float] | None = None
    surface_origin: list[float] | None = None
    surface_tangent_u: list[float] | None = None
    surface_tangent_v: list[float] | None = None
    surface_bounds: dict[str, Any] | None = None
    surface_coordinates: dict[str, Any] | None = None
    surface_binding_source: str | None = None
    patch_id: str | None = None
    editable: bool = True
    position_source: str | None = None
    source: str = "contact_mask"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.start_frame < 0:
            raise ValueError(f"{self.anchor_id}: start_frame must be >= 0")
        if self.end_frame <= self.start_frame:
            raise ValueError(f"{self.anchor_id}: end_frame must be > start_frame for half-open intervals")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ContactPatchRecord:
    motion_id: str
    patch_id: str
    body: str
    start_frame: int
    end_frame: int
    patch_type: str = "unknown"
    patch_center_world: list[float] | None = None
    link_names: list[str] | None = None
    sphere_ids: list[str] | None = None
    anchor_id: str | None = None
    slip_score: float | None = None

    # Stable Newton/MJWarp contact identity. Runtime body/shape integer IDs are
    # intentionally not persisted because they can change when the scene is rebuilt.
    newton_body_label: str | None = None
    newton_shape_labels: list[str] | None = None
    robot_points_local: list[list[float]] | None = None
    robot_normals_local: list[list[float]] | None = None
    source_target_frames: list[int] | None = None
    source_target_points_w: list[list[list[float]]] | None = None
    source_points_local_by_frame: list[list[list[float]]] | None = None
    robot_asset_fingerprint: str | None = None
    robot_binding_backend: str | None = None
    robot_binding_source: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.start_frame < 0:
            raise ValueError(f"{self.patch_id}: start_frame must be >= 0")
        if self.end_frame <= self.start_frame:
            raise ValueError(f"{self.patch_id}: end_frame must be > start_frame")
        if self.patch_center_world is not None and len(self.patch_center_world) != 3:
            raise ValueError(f"{self.patch_id}: patch_center_world must have length 3")
        if self.robot_points_local is not None:
            for index, point in enumerate(self.robot_points_local):
                if len(point) != 3:
                    raise ValueError(f"{self.patch_id}: robot_points_local[{index}] must have length 3")
        if self.robot_normals_local is not None:
            if self.robot_points_local is None:
                raise ValueError(f"{self.patch_id}: robot_normals_local requires robot_points_local")
            if len(self.robot_normals_local) != len(self.robot_points_local):
                raise ValueError(
                    f"{self.patch_id}: robot_normals_local count must match robot_points_local count"
                )
            for index, normal in enumerate(self.robot_normals_local):
                if len(normal) != 3:
                    raise ValueError(f"{self.patch_id}: robot_normals_local[{index}] must have length 3")
        if self.newton_shape_labels is not None and self.robot_points_local is not None:
            if len(self.newton_shape_labels) != len(self.robot_points_local):
                raise ValueError(
                    f"{self.patch_id}: newton_shape_labels count must match robot_points_local count"
                )
        if self.source_target_points_w is not None:
            if self.source_target_frames is None:
                raise ValueError(
                    f"{self.patch_id}: source_target_points_w requires source_target_frames"
                )
            if self.robot_points_local is None:
                raise ValueError(
                    f"{self.patch_id}: source_target_points_w requires robot_points_local"
                )
            if len(self.source_target_points_w) != len(self.source_target_frames):
                raise ValueError(
                    f"{self.patch_id}: source target frame/trajectory counts must match"
                )
            if any(
                len(points) != len(self.robot_points_local)
                or any(len(point) != 3 for point in points)
                for points in self.source_target_points_w
            ):
                raise ValueError(
                    f"{self.patch_id}: source_target_points_w must have shape [F,P,3]"
                )
            if any(
                int(self.source_target_frames[index + 1])
                <= int(self.source_target_frames[index])
                for index in range(len(self.source_target_frames) - 1)
            ):
                raise ValueError(
                    f"{self.patch_id}: source_target_frames must be strictly increasing"
                )
        if self.source_points_local_by_frame is not None:
            if self.source_target_frames is None:
                raise ValueError(
                    f"{self.patch_id}: source_points_local_by_frame requires source_target_frames"
                )
            if self.robot_points_local is None:
                raise ValueError(
                    f"{self.patch_id}: source_points_local_by_frame requires robot_points_local"
                )
            if len(self.source_points_local_by_frame) != len(self.source_target_frames):
                raise ValueError(
                    f"{self.patch_id}: local point frame/trajectory counts must match"
                )
            if any(
                len(points) != len(self.robot_points_local)
                or any(len(point) != 3 for point in points)
                for points in self.source_points_local_by_frame
            ):
                raise ValueError(
                    f"{self.patch_id}: source_points_local_by_frame must have shape [F,P,3]"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ContactAnchorEditRecord:
    """Anchor edit whose ``affected_frames`` interval is half-open ``[start, end)``."""

    edit_id: str
    motion_id: str
    anchor_id: str
    body: str
    edit_type: str = "move_contact_anchor"
    old_world_position: list[float] | None = None
    new_world_position: list[float] | None = None
    requested_delta_world: list[float] | None = None
    delta_world: list[float] | None = None
    tangent_delta: list[float] | None = None
    delta_object: list[float] | None = None
    affected_frames: list[int] | None = None
    surface_id: str | None = None
    surface_normal: list[float] | None = None
    surface_coordinates_before: dict[str, Any] | None = None
    surface_coordinates_after: dict[str, Any] | None = None
    constraint_mode: str | None = None
    clamped: bool = False
    source: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.edit_type != "move_contact_anchor":
            raise ValueError(f"{self.edit_id}: unsupported edit_type {self.edit_type}")
        if self.new_world_position is not None and len(self.new_world_position) != 3:
            raise ValueError(f"{self.edit_id}: new_world_position must have length 3")
        if self.old_world_position is not None and len(self.old_world_position) != 3:
            raise ValueError(f"{self.edit_id}: old_world_position must have length 3")
        if self.delta_world is not None and len(self.delta_world) != 3:
            raise ValueError(f"{self.edit_id}: delta_world must have length 3")
        if self.requested_delta_world is not None and len(self.requested_delta_world) != 3:
            raise ValueError(f"{self.edit_id}: requested_delta_world must have length 3")
        if self.tangent_delta is not None and len(self.tangent_delta) != 2:
            raise ValueError(f"{self.edit_id}: tangent_delta must have length 2")
        if self.delta_object is not None and len(self.delta_object) != 3:
            raise ValueError(f"{self.edit_id}: delta_object must have length 3")
        if self.surface_normal is not None and len(self.surface_normal) != 3:
            raise ValueError(f"{self.edit_id}: surface_normal must have length 3")
        if self.affected_frames is not None:
            if len(self.affected_frames) != 2:
                raise ValueError(f"{self.edit_id}: affected_frames must be [start_frame, end_frame)")
            start, end = int(self.affected_frames[0]), int(self.affected_frames[1])
            if start < 0:
                raise ValueError(f"{self.edit_id}: affected_frames start must be >= 0")
            if end <= start:
                raise ValueError(f"{self.edit_id}: affected_frames must be a non-empty half-open interval")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PoseEditRecord:
    """Task-space pose translation over a half-open frame interval."""

    edit_id: str
    motion_id: str
    affected_frames: list[int]
    translation_world: list[float]
    semantic_names: list[str]
    edit_type: str = "translate_pose"
    weight_scale: float = 1.0
    source: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.edit_type != "translate_pose":
            raise ValueError(f"{self.edit_id}: unsupported pose edit type {self.edit_type!r}")
        if len(self.affected_frames) != 2:
            raise ValueError(f"{self.edit_id}: affected_frames must be [start_frame, end_frame)")
        start, end = (int(value) for value in self.affected_frames)
        if start < 0 or end <= start:
            raise ValueError(f"{self.edit_id}: affected_frames must be a non-empty half-open interval")
        if len(self.translation_world) != 3:
            raise ValueError(f"{self.edit_id}: translation_world must have length 3")
        if not all(math.isfinite(float(value)) for value in self.translation_world):
            raise ValueError(f"{self.edit_id}: translation_world must be finite")
        if not self.semantic_names or any(not str(name) for name in self.semantic_names):
            raise ValueError(f"{self.edit_id}: semantic_names must be non-empty")
        if not math.isfinite(float(self.weight_scale)) or float(self.weight_scale) <= 0.0:
            raise ValueError(f"{self.edit_id}: weight_scale must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ContactTransitionRecord:
    motion_id: str
    transition_id: str
    start_frame: int
    end_frame: int
    active_body: str | None = None
    support_bodies: list[str] = field(default_factory=list)
    start_event_id: str | None = None
    end_event_id: str | None = None
    source_anchor_id: str | None = None
    target_anchor_id: str | None = None
    transition_type: ContactTransitionType = "unknown"
    source: str = "contact_mask"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.start_frame < 0:
            raise ValueError(f"{self.transition_id}: start_frame must be >= 0")
        if self.end_frame <= self.start_frame:
            raise ValueError(f"{self.transition_id}: end_frame must be > start_frame")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
