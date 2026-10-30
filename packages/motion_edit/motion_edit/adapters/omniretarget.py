from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OmniRetargetPaths:
    motion_path: Path
    dataset_root: Path | None
    terrain_urdf: Path | None
    terrain_obj: Path | None
    contact_force_npz: Path | None
    holosoma_motion_path: Path | None


def _find_dataset_root(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        if parent.name == "OmniRetarget_Dataset":
            return parent
    return None


def _climb_id(path: Path) -> int | None:
    match = re.search(r"climb_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def detect_omniretarget_paths(
    motion_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> OmniRetargetPaths:
    motion = Path(motion_path).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve() if repo_root else None
    dataset = Path(dataset_root).expanduser().resolve() if dataset_root else _find_dataset_root(motion)
    cid = _climb_id(motion)

    terrain_urdf = None
    terrain_obj = None
    if cid is not None:
        candidates_urdf = []
        candidates_obj = []
        if dataset is not None:
            candidates_urdf.append(dataset / "models" / "terrain" / f"climb_{cid:02d}" / "multi_boxes_z_scale_1.0.urdf")
        if repo is not None:
            candidates_urdf.append(repo / "OmniRetarget_Dataset" / "models" / "terrain" / f"climb_{cid:02d}" / "multi_boxes_z_scale_1.0.urdf")
            candidates_obj.append(repo / "configs" / "motion_matched" / "terrain_obj_cache" / f"climb_{cid:02d}" / "multi_boxes_z_scale_1.0.obj")
        terrain_urdf = _existing(*candidates_urdf)
        terrain_obj = _existing(*candidates_obj)

    contact_force = None
    if cid is not None and repo is not None:
        contact_force = _existing(
            repo / "data" / "rollout_ref_contact_force_demos" / f"climb_{cid:02d}_rollout_ref_contact_force.npz",
            repo / "data" / "contact_force_demos" / f"climb_{cid:02d}_z_scale_1.0_force_demo_from_29motion_model16000.npz",
        )

    holosoma_motion = None
    if dataset is not None:
        holosoma_motion = _existing(
            dataset / "data" / "holosoma_motions_50hz" / motion.name,
            dataset / "data" / "holosoma_motions_masked_50hz" / motion.name,
        )

    return OmniRetargetPaths(
        motion_path=motion,
        dataset_root=dataset,
        terrain_urdf=terrain_urdf,
        terrain_obj=terrain_obj,
        contact_force_npz=contact_force,
        holosoma_motion_path=holosoma_motion,
    )


def load_motion_matched_manifest(path: str | Path) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)
