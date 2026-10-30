"""Replay the historical full-route sampled-coverage experiment.

Migrated from tmp/legacy_camera_route_coverage_20261013/run.py. The sampled
area and first-hit pixel calculations are preserved for comparison. Inputs
and outputs are explicit; contact labels remain historical diagnostic
targets, not newly certified Newton contact truth.
"""

import json, pathlib, math, time, hashlib, csv, xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial import cKDTree
from somaforge_core.world_mesh_scan import DepthWorldMeshScanner
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from somaforge_core.robot_assets import decode_robot_asset_json, canonical_g1_urdf_path, encode_robot_asset_json
from somaforge_core.contact_face_selection import FACE_POLICY


def run_route(
    *, output, history, profile, motion_index, training_manifest, scene_metadata=None, coverage_distance_m=0.04
):
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    OUT = pathlib.Path(output).resolve()
    if not OUT.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("Outputs must be inside the project tmp/")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Choose an empty --output directory: {OUT}")
    if not math.isfinite(coverage_distance_m) or coverage_distance_m <= 0:
        raise ValueError("Coverage distance must be positive and finite")
    OUT.mkdir(parents=True, exist_ok=True)
    history = pathlib.Path(history).resolve()
    torch.set_num_threads(2)
    manifest = json.loads(pathlib.Path(training_manifest).read_text())
    if not 0 <= motion_index < len(manifest["motion_files"]):
        raise ValueError("Motion index is outside the manifest")
    entry = manifest["motion_files"][motion_index]
    with np.load(entry["motion_file"], allow_pickle=False) as archive:
        decode_robot_asset_json(archive["robot_asset_json"], context="legacy route coverage native motion")
        q = np.asarray(archive["joint_pos"], dtype=np.float32)
        fps = float(archive["fps"])
    source = (
        pathlib.Path(scene_metadata)
        if scene_metadata is not None
        else history / "legacy_camera_route_coverage_20261013/scene_metadata.json"
    )
    metadata = json.loads(source.read_text())
    scanner = DepthWorldMeshScanner(metadata, profile=profile)
    fk = CanonicalG1ForwardKinematics()
    frames = len(q)
    if frames > np.iinfo(np.int16).max or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame count/fps for the historical binary viewer")
    pixel_count = scanner.width * scanner.height
    rows = [
        r
        for r in json.loads((history / "predictor_establish_visibility_20261013/report.json").read_text())["rows"]
        if r["motion_index"] == motion_index
    ]
    if not rows:
        raise ValueError("No historical targets for this motion")
    if any(r["motion_name"] != entry["motion_id"] for r in rows):
        raise ValueError("Motion index/name differs from historical target evidence")
    contracts = json.loads((history / "predictor_wide_depth_20261013/report.json").read_text())["contracts"]
    if any(contracts[str(r["height"])]["geometry_sha256"] != scanner.geometry_sha256 for r in rows):
        raise ValueError("Scene geometry differs from historical targets; supply matching --scene-metadata")
    if any(not 0 <= r["current_frame"] < r["target_frame"] < frames for r in rows):
        raise ValueError("Invalid target sample timing")
    upward = set(scanner.upward_surface_ids)
    known_surfaces = {int(face["surface"]) for face in metadata["surface_catalog"]}
    if any(row["surface"] not in known_surfaces for row in rows):
        raise ValueError("Historical target has an unknown primary surface")
    excluded_targets = [r for r in rows if r["surface"] not in upward]
    rows = [r for r in rows if r["surface"] in upward]
    if not rows:
        raise ValueError("No targets on eligible upward primary surfaces")
    target_points = np.array([r["target_world"] for r in rows])
    target_sf = np.array([r["surface"] for r in rows])
    nearest = np.full((frames, len(rows)), np.inf)
    cell = 0.02
    radius = 3.0
    root_tree = cKDTree(q[:, :2])
    catalog = sorted(metadata["surface_catalog"], key=lambda x: int(x["surface"]))
    surface_info = []
    grids = {}
    all_points = []
    all_sf = []
    next_id = 0
    up = [f for f in catalog if int(f["surface"]) in upward]
    ground_z = min(np.array(f["triangles_w"])[..., 2].mean() for f in up)
    top_triangles = np.concatenate(
        [np.array(f["triangles_w"]) for f in up if np.array(f["triangles_w"])[..., 2].mean() > ground_z + 1e-6]
    )

    def inside_triangles(uv, tri):
        valid = np.zeros(len(uv), bool)
        for t in tri:
            a, b, c = t
            v0 = b - a
            v1 = c - a
            v2 = uv - a
            det = v0[0] * v1[1] - v0[1] * v1[0]
            if abs(det) < 1e-12:
                continue
            x = (v2[:, 0] * v1[1] - v2[:, 1] * v1[0]) / det
            y = (v0[0] * v2[:, 1] - v0[1] * v2[:, 0]) / det
            valid |= (x >= -1e-8) & (y >= -1e-8) & (x + y <= 1 + 1e-8)
        return valid

    for face in catalog:
        sid = int(face["surface"])
        tri = np.array(face["triangles_w"])
        n = np.array(face["normal_w"])
        area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1).sum() / 2)
        ground = sid in upward and abs(tri[..., 2].mean() - ground_z) < 1e-6
        bottom = n[2] < -0.99
        slab = (not ground) and tri[..., 2].max() <= ground_z + 1e-6 and (area > 20 or abs(n[2]) < 0.01)
        name = (
            "地面"
            if ground
            else "箱顶"
            if sid in upward
            else "地基底面"
            if bottom and area > 20
            else "箱底"
            if bottom
            else "地基外缘"
            if slab
            else "箱体侧面"
        ) + f" S{sid}"
        info = dict(
            surface=sid,
            name=name,
            normal=n.tolist(),
            mesh_area_m2=area,
            scope="route XY within 3m; excludes ground under box"
            if ground
            else "structural slab boundary, separately audited"
            if slab
            else "entire face",
            included_in_grid=not slab,
        )
        if not slab:
            u = np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.99 else np.cross([0.0, 0.0, 1.0], n)
            u /= np.linalg.norm(u)
            v = np.array([0.0, 1.0, 0.0]) if abs(n[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
            origin = tri[0, 0]
            basis = np.stack((u, v), -1)
            uvtri = (tri - origin) @ basis
            lo = np.floor(uvtri.min((0, 1)) / cell).astype(int)
            hi = np.ceil(uvtri.max((0, 1)) / cell).astype(int)
            nu, nv = hi - lo
            xx, yy = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
            uv = np.stack((xx.ravel() + 0.5, yy.ravel() + 0.5), -1) * cell
            points = origin + uv @ basis.T
            valid = inside_triangles(uv, uvtri)
            if ground:
                valid &= root_tree.query(points[:, :2])[0] <= radius
                buried = inside_triangles(points[:, :2], top_triangles[..., :2])
                info["excluded_under_box_grid_cells"] = int((valid & buried).sum())
                valid &= ~buried
            table = np.full(len(uv), -1, np.int32)
            count = int(valid.sum())
            table[valid] = np.arange(next_id, next_id + count)
            grids[sid] = (origin, basis, lo, int(nu), int(nv), table)
            all_points.append(points[valid])
            all_sf.extend([sid] * count)
            next_id += count
            info["grid_cells"] = count
            info["grid_area_m2"] = count * cell**2
        surface_info.append(info)
    points = np.concatenate(all_points).astype(np.float32)
    sf = np.array(all_sf, np.int16)
    first = np.full(len(points), -1, np.int16)
    depth = np.zeros((frames, scanner.height, scanner.width), np.float32)
    surface = np.full((frames, pixel_count), -1, np.int16)
    origins = np.zeros((frames, 3), np.float32)
    rotations = np.zeros((frames, 3, 3), np.float32)
    hits = np.zeros((frames, len(catalog)), np.int32)
    surface_column = {int(f["surface"]): i for i, f in enumerate(catalog)}
    pixel_totals = {int(f["surface"]): 0 for f in catalog}
    started = time.time()
    for start in range(0, frames, 8):
        packet = scanner.scan(torch.from_numpy(q[start : start + 8]), fk)
        raw = packet["depth_valid"].numpy().reshape(-1, pixel_count)
        pw = packet["scan_points_world"].numpy()
        ps = packet["scan_surface"].numpy()
        depth[start : start + len(raw)] = packet["depth_image"].numpy()
        surface[start : start + len(raw)] = ps
        origins[start : start + len(raw)] = packet["depth_camera_position_world"].numpy()
        rotations[start : start + len(raw)] = packet["depth_camera_rotation_world"].numpy()
        for b in range(len(raw)):
            frame = start + b
            for sid in pixel_totals:
                mask = raw[b] & (ps[b] == sid)
                cloud = pw[b, mask]
                pixel_totals[sid] += len(cloud)
                hits[frame, surface_column[sid]] = len(cloud)
                if not len(cloud):
                    continue
                targets = np.flatnonzero(target_sf == sid)
                if len(targets):
                    nearest[frame, targets] = np.linalg.norm(
                        cloud[:, None] - target_points[None, targets], axis=-1
                    ).min(0)
                if sid not in grids:
                    continue
                origin, basis, lo, nu, nv, table = grids[sid]
                ij = np.floor(((cloud - origin) @ basis) / cell).astype(int) - lo
                ok = (ij[:, 0] >= 0) & (ij[:, 0] < nu) & (ij[:, 1] >= 0) & (ij[:, 1] < nv)
                idx = table[ij[ok, 1] * nu + ij[ok, 0]]
                idx = np.unique(idx[idx >= 0])
                fresh = idx[first[idx] < 0]
                first[fresh] = frame
        if start % 80 == 0:
            print(
                json.dumps(
                    dict(
                        frame=start,
                        frames=frames,
                        seen_cells=int((first >= 0).sum()),
                        seconds=round(time.time() - started, 1),
                    )
                ),
                flush=True,
            )
    for info in surface_info:
        sid = info["surface"]
        hitframes = np.flatnonzero(hits[:, surface_column[sid]] > 0)
        info.update(
            raw_ray_hits=pixel_totals[sid],
            first_raw_hit_frame=int(hitframes[0]) if len(hitframes) else None,
            frames_with_hits=len(hitframes),
        )
        if info["included_in_grid"]:
            mask = sf == sid
            seen = int((first[mask] >= 0).sum())
            info.update(
                seen_grid_cells=seen, seen_grid_area_m2=seen * cell**2, grid_coverage_fraction=seen / int(mask.sum())
            )
    contact_rows = []
    for i, row in enumerate(rows):
        start = row["current_frame"]
        end = row["target_frame"]
        seen = np.flatnonzero(nearest[:, i] <= coverage_distance_m)
        contact_rows.append(
            dict(
                sample_id=row["sample_id"],
                part=row["part"],
                surface=row["surface"],
                start_frame=start,
                target_frame=end,
                target_world=row["target_world"],
                covered_at_start=bool(nearest[start, i] <= coverage_distance_m),
                covered_frames=int(len(seen)),
                nearest_at_start_cm=float(nearest[start, i] * 100) if np.isfinite(nearest[start, i]) else None,
                best_before_or_at_start_cm=float(nearest[: start + 1, i].min() * 100)
                if np.isfinite(nearest[: start + 1, i]).any()
                else None,
                seen_by_start=bool((nearest[: start + 1, i] <= coverage_distance_m).any()),
                seen_before_target=bool((nearest[:end, i] <= coverage_distance_m).any()),
                seen_anytime=bool(len(seen)),
                first_seen_frame=int(seen[0]) if len(seen) else None,
            )
        )
    robot = ET.parse(canonical_g1_urdf_path()).getroot()
    names = tuple(x.attrib["name"] for x in robot.findall("link"))
    lookup = {n: i for i, n in enumerate(names)}
    edges = [
        [lookup[j.find("parent").attrib["link"]], lookup[j.find("child").attrib["link"]]]
        for j in robot.findall("joint")
    ]
    bones, _ = fk.link_poses(torch.from_numpy(q), names)
    arrays = dict(
        depth=depth,
        surface=surface,
        origins=origins,
        rotations=rotations,
        coverage_points=points,
        coverage_surface=sf,
        first_seen=first,
        bones=bones.numpy().astype(np.float32),
        hits=hits,
    )
    files = {}
    for key, array in arrays.items():
        array = np.ascontiguousarray(array)
        array.tofile(OUT / f"{key}.bin")
        import gzip

        with gzip.open(OUT / f"{key}.bin.gz", "wb") as stream:
            stream.write(array.tobytes())
        files[key] = dict(
            file=f"{key}.bin.gz",
            dtype=str(array.dtype),
            shape=list(array.shape),
            bytes=array.nbytes,
            compressed_bytes=(OUT / f"{key}.bin.gz").stat().st_size,
        )
    report = dict(
        schema="legacy_depth_route_sampled_surface_coverage_v1",
        motion=entry,
        motion_frames=frames,
        fps=fps,
        duration_s=(frames - 1) / fps,
        camera=scanner.contract(),
        metadata_source=str(source),
        robot_asset_json=encode_robot_asset_json(),
        area_method="2cm surface-local grid, counted once when at least one actual first-hit depth pixel lands in a cell; cell-centre boundary inclusion, approximate sampled-area coverage, not continuous full visibility or contact truth",
        ground_scope="Within 3m XY distance of any root position on this route; ground underneath box excluded and counted separately",
        cell_size_m=cell,
        surfaces=surface_info,
        targets=contact_rows,
        target_summary=dict(
            total=len(rows),
            covered_at_start=sum(r["covered_at_start"] for r in contact_rows),
            seen_by_start=sum(r["seen_by_start"] for r in contact_rows),
            seen_before_target=sum(r["seen_before_target"] for r in contact_rows),
            seen_anytime=sum(r["seen_anytime"] for r in contact_rows),
        ),
        elapsed_seconds=time.time() - started,
        device="CPU; torch threads=2; running training untouched",
        sensor_model="Ideal calibrated perspective depth, all terrain sides retained; no body occlusion/no noise",
        coverage_distance_m=coverage_distance_m,
        face_selection=FACE_POLICY,
        excluded_targets=excluded_targets,
        historical_contact_truth_revalidated=False,
        contact_truth=False,
        target_source="imported historical establish targets; not newly materialized contact labels",
        sampling_timing="world pose q[frame] raycast; prefix <= start; before target < target; anytime all motion frames; no solver snapshot sampled here",
        source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (OUT / "data.json").write_text(
        json.dumps(
            dict(
                files=files,
                frames=frames,
                fps=fps,
                width=scanner.width,
                height=scanner.height,
                intrinsics=packet["depth_intrinsics"][0].tolist(),
                root=q[:, :3].tolist(),
                edges=edges,
                terrain=metadata["surface_catalog"],
                surfaces=surface_info,
                targets=contact_rows,
                motion=entry["motion_id"],
                camera=scanner.contract(),
            ),
            ensure_ascii=False,
        )
    )
    with (OUT / "targets.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(contact_rows[0]))
        w.writeheader()
        w.writerows(contact_rows)
    print(
        "FINISHED",
        json.dumps(
            dict(targets=report["target_summary"], seconds=report["elapsed_seconds"], output=str(OUT)),
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Reuse the viewer's highlighted original demonstration, with this run's calibration.
    viewer = history / "legacy_camera_route_coverage_20261013/index.html"
    if motion_index == 144 and viewer.is_file():
        text = viewer.read_text()
        text = text.replace("原始相机走完整条路线：看到过哪些表面？", "相机走完整条路线：看到过哪些表面？")
        text = text.replace(
            "原始躯干相机：87×58，水平89.5°，下俯27°，随躯干旋转。",
            f"相机配置：{scanner.profile}；{scanner.width}×{scanner.height}，水平{scanner.fov:g}°，下俯{scanner.pitch:g}°。",
        )
        text = text.replace("4 cm", f"{coverage_distance_m * 100:g} cm")
        text = text.replace(
            "cameraPoint[2]>=.3&&cameraPoint[2]<=3",
            "cameraPoint[2]>=D.camera.depth_range_m[0]&&cameraPoint[2]<=D.camera.depth_range_m[1]",
        )
        text = text.replace(
            "(d-.3)/2.7", "(d-D.camera.depth_range_m[0])/(D.camera.depth_range_m[1]-D.camera.depth_range_m[0])"
        )
        text = text.replace("有效范围 x:0–86、y:0–57", "有效范围 x:0–${W-1}、y:0–${H-1}")
        text = text.replace("uv[1]>57", "uv[1]>H-1").replace("(uv[1]-57)", "(uv[1]-(H-1))")
        text = text.replace(
            "const canvas=document.getElementById('depth'),ctx=canvas.getContext('2d'),im=ctx.createImageData(W,H);",
            "const canvas=document.getElementById('depth');canvas.width=W;canvas.height=H;const ctx=canvas.getContext('2d'),im=ctx.createImageData(W,H);",
        )
        text = text.replace("原始相机 · 第", "相机 · 第")
        (OUT / "index.html").write_text(text)
    return report
