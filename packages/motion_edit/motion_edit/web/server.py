"""Read-only motion/research playback; trajectory editing is retired."""
from __future__ import annotations

import hashlib
import socket
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from somaforge_core import AssetManifest
from somaforge_core.robot_assets import canonical_g1_urdf_path, somaforge_root

from motion_edit.storage import list_motion_assets, list_motion_versions, read_motion_asset, read_motion_version
from motion_edit.workbench.recent import RecentMotionEntry, clear_recent_motions, read_recent_motions, upsert_recent_motion
from .review import load_review
from .motion_data import load_contact_force_payload

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WEB_DIST = PACKAGE_ROOT / "motion_edit" / "web_dist"


class ReviewRequest(BaseModel):
    path: str
    terrain: str | None = None
    case_index: int = 0
    reference_index: int | None = None


class LoadRequest(BaseModel):
    motion_id: str


@dataclass(frozen=True)
class PlaybackMotion:
    motion_id: str
    motion_asset_id: str
    motion_version_id: str | None
    motion_path: str
    provenance: str


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else somaforge_root() / candidate).resolve()



def _repo_url(path: str | Path) -> str:
    """Map an asset URL without authorizing arbitrary checkout files for serving."""
    manifest = AssetManifest.load(somaforge_root() / "configs/assets_manifest.json")
    return "/repo/" + quote(manifest.logical_path(path).as_posix(), safe="/")


def _matching_force_sidecar(motion_path: Path, sidecar: Path) -> bool:
    """Accept only evidence explicitly bound to this exact source sequence."""
    if motion_path == sidecar:
        return True
    if not sidecar.is_file():
        return False
    with np.load(motion_path, allow_pickle=False) as motion, np.load(sidecar, allow_pickle=False) as force:
        motion_key = "qpos" if "qpos" in motion else "joint_pos"
        count = len(motion[motion_key])
        for name in ("contact_force_part_w", "contact_force_part_mask", "contact_force_part_position_w",
                     "contact_force_part_position_valid", "raw_contact_count", "raw_contact_point0_w", "raw_contact_force_w"):
            if name in force and (force[name].ndim == 0 or len(force[name]) != count):
                return False
        if "fps" in force and not np.array_equal(np.asarray(motion.get("fps", 50.)).reshape(-1),
                                                     np.asarray(force["fps"]).reshape(-1)):
            return False
        if "source_sha256" in force:
            return str(force["source_sha256"].item()) == hashlib.sha256(motion_path.read_bytes()).hexdigest()
        force_key = "qpos" if "qpos" in force else "joint_pos"
        if force_key not in force or "joint_names" not in force:
            return False
        return (np.array_equal(motion[motion_key], force[force_key])
                and np.array_equal(motion["joint_names"], force["joint_names"])
                and np.array_equal(np.asarray(motion.get("fps", 50.)).reshape(-1),
                                   np.asarray(force.get("fps", 50.)).reshape(-1)))


def _registered_annotations(payload: dict, motion: PlaybackMotion, asset) -> None:
    """Keep recorded evidence and historical overlays without enabling editing."""
    motion_path = _repo_path(motion.motion_path)
    force_path = motion_path
    if motion.motion_version_id is None and getattr(asset, "contact_force_npz", None):
        sidecar = _repo_path(asset.contact_force_npz)
        if _matching_force_sidecar(motion_path, sidecar):
            force_path = sidecar
        else:
            payload["review"]["notice"] += " 独立力记录与当前运动不匹配，未附加该记录。"
    with np.load(force_path, allow_pickle=False) as force:
        has_forces = any(key in force for key in ("contact_force_part_w", "raw_contact_force_w", "contact_semantics_json"))
    if has_forces:
        payload["contact_force"] = load_contact_force_payload(
            force_path, frame_count=len(payload["qpos"]), motion_path=motion_path)
    # A generated version must never inherit its source motion's contact layer.
    if motion.motion_version_id is None and getattr(asset, "source_contact_layer", None):
        from motion_edit.contact.layers import read_contact_graph
        from motion_edit.paths import LAYERS_ROOT
        layer = Path(asset.source_contact_layer).expanduser()
        if not layer.is_absolute():
            layer = LAYERS_ROOT / layer
        graph = read_contact_graph(layer, asset.motion_id or asset.motion_asset_id)
        payload["graph"] = graph.to_dict()
        for anchor in payload["graph"]["anchors"]:
            anchor["editable"] = False
        payload["review"]["annotation_status"] = "historical_catalog_annotations"
        payload["review"]["notice"] += " 目录接触图仅作历史标注展示，不替代当前求解器接触证据。"
    if getattr(asset, "surface_catalog_path", None):
        from motion_edit.contact.io import read_contact_surfaces
        payload["surfaces"] = [surface.to_dict() for surface in read_contact_surfaces(_repo_path(asset.surface_catalog_path))]


def _resolve_motion(motion_id: str) -> PlaybackMotion:
    try:
        version = read_motion_version(motion_id)
    except FileNotFoundError:
        asset = read_motion_asset(motion_id)
        return PlaybackMotion(asset.motion_asset_id, asset.motion_asset_id, None, asset.motion_path, "source")
    if not version.motion_asset_id:
        raise ValueError(f"{motion_id} has no Motion context binding")
    asset = read_motion_asset(version.motion_asset_id)
    return PlaybackMotion(version.motion_version_id, asset.motion_asset_id, version.motion_version_id,
                          version.motion_path, str(version.metadata.get("provenance") or version.kind))


def _list_motions() -> list[PlaybackMotion]:
    motions = {asset.motion_asset_id: PlaybackMotion(asset.motion_asset_id, asset.motion_asset_id,
                None, asset.motion_path, "source") for asset in list_motion_assets()}
    for version in list_motion_versions():
        if not version.motion_asset_id:
            continue
        try:
            motion = _resolve_motion(version.motion_version_id)
        except (FileNotFoundError, ValueError):
            continue
        if motion.motion_id in motions:
            raise ValueError(f"duplicate unified Motion ID: {motion.motion_id}")
        motions[motion.motion_id] = motion
    return [motions[key] for key in sorted(motions)]


def create_app(*, initial_motion_id: str | None = None, review_path: str | None = None,
               review_terrain: str | None = None, reset_recent_on_start: bool = False) -> FastAPI:
    app = FastAPI(title="SomaForge Motion Review")
    manifest = AssetManifest.load(somaforge_root() / "configs/assets_manifest.json")
    state = {"payload": None, "motion_id": None}
    files: dict[str, Path] = {}
    robot_root = canonical_g1_urdf_path().resolve().parent
    mesh_suffixes = {".urdf", ".obj", ".stl", ".dae", ".mtl", ".png", ".jpg", ".jpeg", ".glb", ".gltf", ".bin"}

    def asset_url(path: str | Path) -> str:
        target = _repo_path(path)
        logical_path = manifest.logical_path(target)
        if target.suffix.lower() not in mesh_suffixes or any(part.startswith(".") for part in logical_path.parts):
            raise ValueError("playback assets must be robot or terrain mesh files")
        logical = logical_path.as_posix()
        files[logical] = target
        return "/repo/" + quote(logical, safe="/")

    def review_payload(path: str, terrain: str | None = None, *, case_index: int = 0,
                       reference_index: int | None = None) -> dict:
        return load_review(_repo_path(path), url=asset_url,
                           terrain=_repo_path(terrain) if terrain else None,
                           case_index=case_index, reference_index=reference_index)

    def open_review(path: str, terrain: str | None = None, *, case_index: int = 0,
                    reference_index: int | None = None) -> dict:
        payload = review_payload(path, terrain, case_index=case_index, reference_index=reference_index)
        state.update(payload=payload, motion_id=None)
        return payload

    def open_motion(motion_id: str) -> dict:
        motion = _resolve_motion(motion_id)
        asset = read_motion_asset(motion.motion_asset_id)
        payload = review_payload(motion.motion_path, asset.terrain_mesh)
        payload.update(motion_id=motion.motion_id, source_motion_id=asset.motion_id or asset.motion_asset_id,
                       provenance=motion.provenance)
        _registered_annotations(payload, motion, asset)
        upsert_recent_motion(RecentMotionEntry(
            label=motion.motion_id, motion_path=str(_repo_path(motion.motion_path)),
            motion_id=asset.motion_id or asset.motion_asset_id, motion_ref_id=motion.motion_id,
            motion_asset_id=asset.motion_asset_id, motion_version_id=motion.motion_version_id,
            metadata={"provenance": motion.provenance},
        ))
        state.update(payload=payload, motion_id=motion.motion_id)
        return payload

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "mode": "read_only"}

    @app.get("/api/motions")
    def motions() -> dict:
        return {"motions": [dict(motion_id=item.motion_id, label=item.motion_id,
                                provenance=item.provenance, ready=_repo_path(item.motion_path).is_file())
                            for item in _list_motions()]}

    @app.get("/api/recent")
    @app.get("/api/recent-motions")
    def recent() -> dict:
        return {"items": [dict(label=e.label, motion_id=e.motion_ref_id,
                               provenance=e.metadata.get("provenance", "source"),
                               active=e.motion_ref_id == state["motion_id"], last_opened_at=e.last_opened_at)
                          for e in read_recent_motions() if e.motion_ref_id],
                "active_motion_id": state["motion_id"]}

    @app.delete("/api/recent")
    @app.delete("/api/recent-motions")
    def clear_recent() -> dict:
        clear_recent_motions()
        return recent()

    @app.post("/api/review/load")
    def review_load(request: ReviewRequest) -> dict:
        try:
            return open_review(request.path, request.terrain, case_index=request.case_index,
                               reference_index=request.reference_index)
        except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/session/load")
    def motion_load(request: LoadRequest) -> dict:
        try:
            return open_motion(request.motion_id)
        except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/session")
    def session() -> dict:
        if state["payload"] is None:
            raise HTTPException(status_code=404, detail="no motion is loaded")
        return state["payload"]

    @app.get("/repo/{asset_path:path}")
    def asset_file(asset_path: str) -> FileResponse:
        # Serve selected assets and canonical robot meshes, never the checkout.
        logical = Path(asset_path)
        if logical.is_absolute() or ".." in logical.parts:
            raise HTTPException(status_code=404, detail="asset not found")
        candidate = files.get(asset_path)
        if candidate is None:
            resolved = (manifest.root / logical).resolve()
            if resolved.is_relative_to(robot_root) and resolved.suffix.lower() in mesh_suffixes:
                candidate = resolved
        if candidate is None or not candidate.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(candidate)

    @app.get("/")
    def index() -> FileResponse:
        target = WEB_DIST / "index.html"
        if not target.is_file():
            raise HTTPException(status_code=503, detail="web frontend is not built; run npm run build in packages/motion_edit/web")
        return FileResponse(target)

    if (WEB_DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")), name="web-assets")
    if reset_recent_on_start:
        clear_recent_motions()
    if review_path:
        open_review(review_path, review_terrain)
    elif initial_motion_id:
        open_motion(initial_motion_id)
    return app


def _port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, int(port))) != 0


def run_review(*, motion_id: str | None = None, review_path: str | None = None,
               review_terrain: str | None = None, host: str = "127.0.0.1", port: int = 8094,
               open_browser: bool = True) -> None:
    import uvicorn

    if not _port_available(port, host):
        raise RuntimeError(f"motion review port {port} is already in use")
    app = create_app(initial_motion_id=motion_id, review_path=review_path, review_terrain=review_terrain)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"SomaForge Motion Review: {url}")
    uvicorn.run(app, host=host, port=port, log_level="info")


# The old launcher remains a read-only alias for existing playback scripts.
run_contact_editor = run_review
