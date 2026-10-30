from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MotionVersionKind = Literal["raw", "augmented"]
TokenStatus = Literal["candidate", "accepted", "rejected", "manual"]


@dataclass(frozen=True)
class MotionAssetRecord:
    motion_asset_id: str
    motion_path: str
    source: str = "local"
    fps: float | None = None
    motion_id: str | None = None
    terrain_id: str | None = None
    terrain_urdf: str | None = None
    terrain_mesh: str | None = None
    surface_catalog_path: str | None = None
    source_motion_path: str | None = None
    contact_force_npz: str | None = None
    source_manifest: str | None = None
    asset_hashes: dict[str, str] = field(default_factory=dict)
    contact_layer: str | None = None
    bound_contact_layer: str | None = None
    edit_plan_path: str | None = None
    output_contact_layer: str | None = None
    output_segment_layer: str | None = None
    raw_contact: dict[str, Any] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep old ``derived`` bundle metadata and new top-level fields in sync.

        Older assets stored contact/editor fields under ``derived``. The contact
        editor now treats a MotionAsset as the complete load bundle, so these
        fields also live at the top level. This shim keeps both representations
        readable while making newly written assets self-contained.
        """

        derived = dict(self.derived or {})
        mappings = {
            "contact_layer": "contact_layer",
            "bound_contact_layer": "bound_contact_layer",
            "edit_plan_path": "edit_plan_path",
            "output_contact_layer": "output_contact_layer",
            "output_segment_layer": "output_segment_layer",
        }
        for attr, key in mappings.items():
            value = getattr(self, attr)
            if value is None and derived.get(key):
                object.__setattr__(self, attr, str(derived[key]))
            elif value is not None and not derived.get(key):
                derived[key] = value
        object.__setattr__(self, "derived", derived)

    @property
    def source_contact_layer(self) -> str | None:
        return self.bound_contact_layer or self.contact_layer

    def missing_contact_editor_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.motion_path:
            missing.append("motion_path")
        if not (self.motion_id or self.motion_asset_id):
            missing.append("motion_id")
        if not self.source_contact_layer:
            missing.append("contact_layer")
        if not (self.surface_catalog_path or self.terrain_mesh or self.terrain_urdf):
            missing.append("surface_catalog_path_or_terrain_asset")
        return missing

    def validate(self) -> None:
        if not self.motion_asset_id:
            raise ValueError("motion_asset_id is required")
        if not self.motion_path:
            raise ValueError(f"{self.motion_asset_id}: motion_path is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class MotionVersionRecord:
    motion_version_id: str
    motion_path: str
    kind: MotionVersionKind = "raw"
    base_motion_id: str | None = None
    motion_asset_id: str | None = None
    parent_motion_version_id: str | None = None
    contact_layer: str | None = None
    canonical_segment_path: str | None = None
    token_catalog_path: str | None = None
    edit_plan_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.motion_version_id:
            raise ValueError("motion_version_id is required")
        if not self.motion_path:
            raise ValueError(f"{self.motion_version_id}: motion_path is required")
        if self.kind not in {"raw", "augmented"}:
            raise ValueError(f"{self.motion_version_id}: unsupported kind {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SegmentIndexManifest:
    motion_version_id: str
    segment_path: str
    segment_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.motion_version_id:
            raise ValueError("motion_version_id is required")
        if self.segment_count < 0:
            raise ValueError(f"{self.motion_version_id}: segment_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TokenRecord:
    token_id: str
    motion_version_id: str
    segment_id: str
    token_family: str
    active_body: str | None = None
    support_bodies: list[str] = field(default_factory=list)
    source_anchor_id: str | None = None
    target_anchor_id: str | None = None
    parent_transition_id: str | None = None
    continuous_params: dict[str, Any] = field(default_factory=dict)
    status: TokenStatus = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.token_id:
            raise ValueError("token_id is required")
        if not self.motion_version_id:
            raise ValueError(f"{self.token_id}: motion_version_id is required")
        if not self.segment_id:
            raise ValueError(f"{self.token_id}: segment_id is required")
        if not self.token_family:
            raise ValueError(f"{self.token_id}: token_family is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
