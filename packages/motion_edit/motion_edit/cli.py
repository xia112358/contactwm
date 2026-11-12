"""Shared motion catalog/data tools and read-only experiment playback."""

from __future__ import annotations

import argparse

from dataclasses import replace

from pathlib import Path

from .adapters.omniretarget import detect_omniretarget_paths

from .adapters.asset_manifest import load_asset_manifest

from .curation import filter_segments

from .export import export_contact_overlay, export_motion_version_manifest, export_split_npz, export_surface_binding_overlay, export_surface_binding_report

from .force_proto import contact_graph_from_masked_motion, segments_from_masked_motion

from .io import read_jsonl

from .layers import read_layer, write_layer

from .paths import EXPORTS_ROOT, LAYERS_ROOT, SURFACES_ROOT, ensure_data_dirs, layer_dir

from .contact import bind_anchors_to_surfaces, bind_segment_to_contact_graph, filter_short_raw_missing_anchors, read_contact_surfaces, refine_contact_graph_anchor_positions_from_raw_contacts, split_foot_contact_anchors, write_contact_layer, write_contact_surfaces

from .contact.graph import ContactGraph

from .contact.layers import read_contact_graph

from .contact.patches import patches_from_anchors

from .contact.surface_catalog import box_surfaces, parse_box_descriptor, surfaces_from_obj_mesh_faces, surfaces_from_urdf_meshes

from .storage.canonical import build_canonical_segments, mark_canonical_segment_statuses, write_motion_version_with_canonical_segments

from .storage.io import list_motion_assets, read_canonical_segments, read_motion_asset, read_motion_version, read_token_catalog, replace_canonical_segments, write_motion_asset, write_motion_version, write_token_catalog

from .storage.schema import MotionAssetRecord, MotionVersionRecord

from .storage.tokens import build_tokens_from_segments

from .workbench import replace_segment, select_segment, split_segment, trim_segment

def _cmd_init(_args: argparse.Namespace) -> None:
    ensure_data_dirs()
    print(f"initialized {LAYERS_ROOT.parents[0]}")

def _cmd_import_force_proto(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    motion_dir = Path(args.motion_dir).expanduser().resolve()
    out_dir = layer_dir("candidate", args.layer_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    registered_by_path = {}
    if getattr(args, "use_registered_motion_ids", False) or getattr(args, "update_motions", False):
        registered_by_path = {
            str(Path(record.motion_path).expanduser().resolve()): record
            for record in list_motion_assets()
        }
    motion_paths = sorted(motion_dir.glob(args.pattern))
    explicit_motion_id = getattr(args, "motion_id", None)
    if explicit_motion_id and len(motion_paths) != 1:
        raise ValueError("--motion-id requires --pattern to match exactly one motion")
    total = 0
    for motion_path in motion_paths:
        registered = registered_by_path.get(str(motion_path.resolve()))
        motion_id = explicit_motion_id or (
            registered.motion_id
            if registered is not None and registered.motion_id
            else None
        )
        segments = segments_from_masked_motion(motion_path, source=args.source, status="candidate", motion_id=motion_id)
        write_layer(out_dir / f"{motion_path.stem}.jsonl", segments)
        graph = contact_graph_from_masked_motion(motion_path, source=args.source, motion_id=motion_id)
        surfaces = None
        surface_catalog = getattr(args, "surface_catalog", None)
        max_surface_distance = float(
            getattr(args, "max_surface_distance", 0.08)
        )
        if surface_catalog:
            surfaces = read_contact_surfaces(Path(surface_catalog).expanduser())
            graph = refine_contact_graph_anchor_positions_from_raw_contacts(
                graph,
                motion_path,
                surfaces=surfaces,
                max_surface_distance=max_surface_distance,
            )
            graph = split_foot_contact_anchors(graph)
            graph, _filter_events = filter_short_raw_missing_anchors(
                graph,
                drop_classes={
                    "raw_missing",
                    "edge_candidate",
                    "outside_known_surfaces",
                },
                source="import_force_proto_surface_filter",
            )
            bound_anchors = bind_anchors_to_surfaces(
                graph.anchors,
                surfaces,
                max_distance=max_surface_distance,
                mode="reject",
            )
            graph = ContactGraph(
                motion_id=graph.motion_id,
                events=graph.events,
                anchors=bound_anchors,
                patches=patches_from_anchors(bound_anchors),
                transitions=graph.transitions,
            )
            counts = _surface_binding_counts(graph)
            if counts["unbound_count"] or counts["failed_count"]:
                raise ValueError(
                    "surface-aware force proto import requires every retained "
                    "contact anchor to be bound: "
                    f"anchors={counts['anchor_count']} "
                    f"bound={counts['bound_count']} "
                    f"unbound={counts['unbound_count']} "
                    f"failed={counts['failed_count']}"
                )
        write_contact_layer(LAYERS_ROOT / "contact" / args.layer_name, graph)
        if surfaces is not None:
            write_contact_surfaces(
                LAYERS_ROOT
                / "contact"
                / args.layer_name
                / "surfaces"
                / f"{graph.motion_id}.jsonl",
                surfaces,
            )
        if registered is not None and getattr(args, "update_motions", False):
            derived = dict(registered.derived or {})
            derived["contact_layer"] = f"contact/{args.layer_name}"
            write_motion_asset(replace(registered, contact_layer=f"contact/{args.layer_name}", derived=derived))
        total += len(segments)
    print(f"wrote {total} candidate segments to {out_dir}; contact layer={LAYERS_ROOT / 'contact' / args.layer_name}")

def _cmd_import_asset_manifest(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    selected = set(args.motion_id or [])
    assets = load_asset_manifest(args.manifest, verify_hashes=args.verify_hashes)
    if selected:
        assets = [asset for asset in assets if asset.motion_id in selected or str(asset.motion_index) in selected]
    candidate_root = layer_dir("candidate", args.layer_name)
    candidate_root.mkdir(parents=True, exist_ok=True)
    contact_layer = f"contact/{args.layer_name}"
    segment_count = 0
    for asset in assets:
        surfaces = surfaces_from_obj_mesh_faces(
            motion_id=asset.motion_id,
            obj_path=asset.terrain_mesh_path,
            include_sides=False,
            include_ground=True,
        )
        surface_path = SURFACES_ROOT / f"{asset.motion_id}_terrain_surfaces.jsonl"
        write_contact_surfaces(surface_path, surfaces)
        segments = segments_from_masked_motion(
            asset.force_motion_path,
            source=args.layer_name,
            status="candidate",
            motion_id=asset.motion_id,
        )
        write_layer(candidate_root / f"{asset.motion_id}.jsonl", segments)
        graph = contact_graph_from_masked_motion(
            asset.force_motion_path,
            source=args.layer_name,
            motion_id=asset.motion_id,
        )
        write_contact_layer(LAYERS_ROOT / contact_layer, graph)
        record = MotionAssetRecord(
            motion_asset_id=asset.motion_asset_id,
            motion_path=str(asset.reference_motion_path),
            source="newton_asset_manifest",
            fps=args.fps,
            motion_id=asset.motion_id,
            terrain_id=str(asset.motion_index),
            terrain_mesh=str(asset.terrain_mesh_path),
            surface_catalog_path=str(surface_path),
            source_motion_path=str(asset.source_motion_path),
            contact_force_npz=str(asset.force_motion_path),
            source_manifest=str(asset.manifest_path),
            asset_hashes={
                "reference_motion_sha256": asset.reference_motion_sha256,
                "motion_sha256": asset.motion_sha256,
                "source_sha256": asset.source_sha256,
                "terrain_sha256": asset.terrain_sha256,
                "contact_solver_sha256": asset.contact_solver_sha256,
            },
            contact_layer=contact_layer,
            raw_contact={"available": True, "source": "newton_8part"},
            metadata={
                "manifest_motion_index": asset.motion_index,
                "kinematic_reference": "clean_reference_motion_file",
                "contact_force_source": "newton_rollout_motion_file",
                "contact_force_provenance": asset.provenance,
            },
        )
        write_motion_asset(record)
        segment_count += len(segments)
    print(
        f"imported {len(assets)} manifest assets, {segment_count} candidate segments; "
        f"contact layer={LAYERS_ROOT / contact_layer}"
    )

def _cmd_export_contact_overlay(args: argparse.Namespace) -> None:
    graph = read_contact_graph(LAYERS_ROOT / args.source, args.motion_id)
    output = export_contact_overlay(args.output, graph)
    print(f"wrote contact overlay {output}")

def _read_surfaces_for_contact_layer(contact_layer: str, motion_id: str, surface_catalog: str | None):
    if surface_catalog:
        return read_contact_surfaces(Path(surface_catalog).expanduser())
    sidecar = LAYERS_ROOT / contact_layer / "surfaces" / f"{motion_id}.jsonl"
    if sidecar.exists():
        return read_contact_surfaces(sidecar)
    return []

def _surface_binding_counts(graph) -> dict[str, int]:
    statuses = []
    for anchor in graph.anchors:
        bindings = anchor.metadata.get("surface_bindings")
        latest = bindings[-1] if isinstance(bindings, list) and bindings else None
        if anchor.metadata.get("surface_binding_failed"):
            statuses.append("failed")
        elif not anchor.surface_id:
            statuses.append("unbound")
        elif isinstance(latest, dict) and latest.get("clamped"):
            statuses.append("clamped")
        elif anchor.surface_id and latest is None:
            statuses.append("suspicious")
        else:
            statuses.append("bound")
    return {
        "anchor_count": len(graph.anchors),
        "bound_count": statuses.count("bound") + statuses.count("clamped") + statuses.count("suspicious"),
        "unbound_count": statuses.count("unbound"),
        "failed_count": statuses.count("failed"),
        "clamped_count": statuses.count("clamped"),
        "suspicious_count": statuses.count("suspicious"),
    }

def _cmd_export_surface_binding_report(args: argparse.Namespace) -> None:
    graph = read_contact_graph(LAYERS_ROOT / args.contact_layer, args.motion_id)
    surfaces = _read_surfaces_for_contact_layer(args.contact_layer, args.motion_id, args.surface_catalog)
    out = export_surface_binding_report(args.output, graph=graph, surfaces=surfaces)
    counts = _surface_binding_counts(graph)
    print(
        f"wrote surface binding report {out} anchors={counts['anchor_count']} "
        f"bound={counts['bound_count']} unbound={counts['unbound_count']} failed={counts['failed_count']} "
        f"clamped={counts['clamped_count']} surfaces={len(surfaces)}"
    )

def _cmd_export_surface_binding_overlay(args: argparse.Namespace) -> None:
    graph = read_contact_graph(LAYERS_ROOT / args.contact_layer, args.motion_id)
    surfaces = _read_surfaces_for_contact_layer(args.contact_layer, args.motion_id, args.surface_catalog)
    out = export_surface_binding_overlay(args.output, graph=graph, surfaces=surfaces)
    counts = _surface_binding_counts(graph)
    print(
        f"wrote surface binding overlay {out} anchors={counts['anchor_count']} "
        f"bound={counts['bound_count']} unbound={counts['unbound_count']} failed={counts['failed_count']} "
        f"clamped={counts['clamped_count']} surfaces={len(surfaces)}"
    )

def _cmd_summarize_surface_bindings(args: argparse.Namespace) -> None:
    graph = read_contact_graph(LAYERS_ROOT / args.contact_layer, args.motion_id)
    surfaces = _read_surfaces_for_contact_layer(args.contact_layer, args.motion_id, args.surface_catalog)
    counts = _surface_binding_counts(graph)
    used_surfaces = sorted({anchor.surface_id for anchor in graph.anchors if anchor.surface_id})
    suspicious = []
    for anchor in graph.anchors:
        bindings = anchor.metadata.get("surface_bindings")
        if anchor.metadata.get("surface_binding_failed") or (anchor.surface_id and not bindings):
            suspicious.append(anchor)
    print(f"motion_id={args.motion_id}")
    print(
        f"anchors={counts['anchor_count']} bound={counts['bound_count']} unbound={counts['unbound_count']} "
        f"failed={counts['failed_count']} clamped={counts['clamped_count']} suspicious={counts['suspicious_count']}"
    )
    print(f"surfaces={len(surfaces)} used={','.join(used_surfaces) if used_surfaces else '<none>'}")
    for anchor in suspicious[: args.limit]:
        reason = anchor.metadata.get("surface_binding_failure_reason") or "binding metadata missing"
        print(f"suspicious {anchor.anchor_id} body={anchor.body} surface={anchor.surface_id} reason={reason}")

def _cmd_bind_contact_surfaces(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    contact_layer = args.contact_layer
    version = None
    if args.motion_version_id:
        version = read_motion_version(args.motion_version_id)
        if contact_layer is None:
            contact_layer = version.contact_layer
    if contact_layer is None:
        raise ValueError("bind-contact-surfaces requires --contact-layer or --motion-version-id with contact_layer")
    if args.update_motion_version and not args.motion_version_id:
        raise ValueError("--update-motion-version requires --motion-version-id")
    graph = read_contact_graph(LAYERS_ROOT / contact_layer, args.motion_id)
    surface_catalog = _resolve_surface_catalog(
        motion_id=args.motion_id,
        surface_catalog=args.surface_catalog,
        terrain_urdf=getattr(args, "terrain_urdf", None),
        terrain_mesh=getattr(args, "terrain_mesh", None),
        include_side_surfaces=getattr(args, "include_side_surfaces", False),
        include_ground=not getattr(args, "no_ground", False),
        ground_z=getattr(args, "ground_z", 0.0),
        ground_half_extent=getattr(args, "ground_half_extent", 10.0),
    )
    surfaces = _filter_surfaces_for_binding(
        read_contact_surfaces(surface_catalog),
        include_side_surfaces=getattr(args, "include_side_surfaces", False),
    )
    bound_anchors = bind_anchors_to_surfaces(
        graph.anchors,
        surfaces,
        max_distance=args.max_distance,
        mode=args.mode,
    )
    bound_graph = ContactGraph(
        motion_id=graph.motion_id,
        events=graph.events,
        anchors=bound_anchors,
        patches=patches_from_anchors(bound_anchors),
        transitions=graph.transitions,
    )
    out_layer = write_contact_layer(LAYERS_ROOT / args.output_contact_layer, bound_graph)
    write_contact_surfaces(out_layer / "surfaces" / f"{args.motion_id}.jsonl", surfaces)
    bound_count = sum(1 for anchor in bound_anchors if anchor.surface_id)
    failed_count = sum(1 for anchor in bound_anchors if anchor.metadata.get("surface_binding_failed"))
    clamped_count = sum(
        1
        for anchor in bound_anchors
        for binding in anchor.metadata.get("surface_bindings", [])
        if binding.get("clamped")
    )
    if args.update_motion_version and version is not None:
        write_motion_version(replace(version, contact_layer=args.output_contact_layer))
    if args.rebind_canonical_segments:
        if not args.motion_version_id:
            raise ValueError("--rebind-canonical-segments requires --motion-version-id")
        segments = [
            bind_segment_to_contact_graph(segment, bound_graph)
            for segment in read_canonical_segments(args.motion_version_id)
        ]
        replace_canonical_segments(
            args.motion_version_id,
            segments,
            reason=f"surface binding from {args.output_contact_layer}",
            source="surface_binding",
            kind="surface_binding_rebind",
        )
    print(
        f"bound contact surfaces motion={args.motion_id} total={len(bound_anchors)} "
        f"bound={bound_count} unbound={len(bound_anchors) - bound_count} clamped={clamped_count} failed={failed_count}"
    )
    print(f"wrote contact layer {out_layer}")

def _cmd_create_box_surface_catalog(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    surfaces = []
    for descriptor in args.box:
        object_id, center, size = parse_box_descriptor(descriptor)
        surfaces.extend(
            box_surfaces(
                motion_id=args.motion_id,
                object_id=object_id,
                center=center,
                size=size,
                include_sides=not args.top_only,
            )
        )
    write_contact_surfaces(args.output, surfaces)
    print(f"wrote {len(surfaces)} contact surfaces to {Path(args.output).expanduser()}")

def _default_terrain_surface_catalog(motion_id: str) -> Path:
    return SURFACES_ROOT / f"{motion_id}_terrain_surfaces.jsonl"

def _write_urdf_surface_catalog(
    *,
    motion_id: str,
    terrain_urdf: str | Path,
    output: str | Path | None,
    include_side_surfaces: bool = False,
    include_ground: bool = True,
    ground_z: float = 0.0,
    ground_half_extent: float = 10.0,
) -> Path:
    if output is not None:
        out = Path(output).expanduser()
    elif include_side_surfaces:
        out = SURFACES_ROOT / f"{motion_id}_terrain_surfaces_with_sides.jsonl"
    else:
        out = _default_terrain_surface_catalog(motion_id)
    surfaces = surfaces_from_urdf_meshes(
        motion_id=motion_id,
        urdf_path=terrain_urdf,
        include_sides=include_side_surfaces,
        include_ground=include_ground,
        ground_z=ground_z,
        ground_half_extent=ground_half_extent,
    )
    write_contact_surfaces(out, surfaces)
    return out

def _write_obj_surface_catalog(
    *,
    motion_id: str,
    terrain_mesh: str | Path,
    output: str | Path | None,
    include_side_surfaces: bool = False,
    include_ground: bool = True,
    ground_z: float = 0.0,
    ground_half_extent: float = 10.0,
) -> Path:
    if output is not None:
        out = Path(output).expanduser()
    elif include_side_surfaces:
        out = SURFACES_ROOT / f"{motion_id}_terrain_surfaces_with_sides.jsonl"
    else:
        out = _default_terrain_surface_catalog(motion_id)
    surfaces = surfaces_from_obj_mesh_faces(
        motion_id=motion_id,
        obj_path=terrain_mesh,
        include_sides=include_side_surfaces,
        include_ground=include_ground,
        ground_z=ground_z,
        ground_half_extent=ground_half_extent,
    )
    write_contact_surfaces(out, surfaces)
    return out

def _resolve_surface_catalog(
    *,
    motion_id: str,
    surface_catalog: str | None,
    terrain_urdf: str | Path | None,
    terrain_mesh: str | Path | None = None,
    include_side_surfaces: bool = False,
    include_ground: bool = True,
    ground_z: float = 0.0,
    ground_half_extent: float = 10.0,
) -> Path:
    if surface_catalog:
        path = Path(surface_catalog).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if terrain_urdf:
        out = _write_urdf_surface_catalog(
            motion_id=motion_id,
            terrain_urdf=terrain_urdf,
            output=None,
            include_side_surfaces=include_side_surfaces,
            include_ground=include_ground,
            ground_z=ground_z,
            ground_half_extent=ground_half_extent,
        )
        print(f"generated terrain surface catalog from URDF: {out}")
        return out
    if terrain_mesh:
        out = _write_obj_surface_catalog(
            motion_id=motion_id,
            terrain_mesh=terrain_mesh,
            output=None,
            include_side_surfaces=include_side_surfaces,
            include_ground=include_ground,
            ground_z=ground_z,
            ground_half_extent=ground_half_extent,
        )
        print(f"generated terrain surface catalog from OBJ: {out}")
        return out
    raise ValueError("surface catalog is required; pass --surface-catalog, --terrain-urdf, or --terrain-mesh")

def _filter_surfaces_for_binding(surfaces, *, include_side_surfaces: bool):
    if include_side_surfaces:
        return list(surfaces)
    return [
        surface
        for surface in surfaces
        if surface.surface_id == "terrain_ground_z0" or float(surface.normal[2]) > 0.5
    ]

def _cmd_create_urdf_surface_catalog(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    out = _write_urdf_surface_catalog(
        motion_id=args.motion_id,
        terrain_urdf=args.terrain_urdf,
        output=args.output,
        include_side_surfaces=args.include_side_surfaces,
        include_ground=not args.no_ground,
        ground_z=args.ground_z,
        ground_half_extent=args.ground_half_extent,
    )
    surfaces = read_contact_surfaces(out)
    print(f"wrote {len(surfaces)} URDF contact surfaces to {out}")

def _cmd_register_motion_asset(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    record = MotionAssetRecord(
        motion_asset_id=args.motion_asset_id,
        motion_path=str(Path(args.motion).expanduser()),
        source=args.source,
        fps=args.fps,
        motion_id=getattr(args, "motion_id", None),
        terrain_id=getattr(args, "terrain_id", None),
        terrain_urdf=str(Path(args.terrain_urdf).expanduser()) if getattr(args, "terrain_urdf", None) else None,
        terrain_mesh=str(Path(args.terrain_mesh).expanduser()) if getattr(args, "terrain_mesh", None) else None,
        surface_catalog_path=str(Path(args.surface_catalog).expanduser()) if getattr(args, "surface_catalog", None) else None,
        raw_contact={
            "available": bool(getattr(args, "raw_contact", False)),
            "source": getattr(args, "raw_contact_source", None),
        },
        derived={
            key: value
            for key, value in {
                "contact_layer": getattr(args, "contact_layer", None),
                "bound_contact_layer": getattr(args, "bound_contact_layer", None),
                "edit_plan_path": getattr(args, "edit_plan", None),
                "output_contact_layer": getattr(args, "output_contact_layer", None),
            }.items()
            if value
        },
    )
    out = write_motion_asset(record)
    print(f"registered motion {record.motion_asset_id} path={out}")

def _add_register_motion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motion-asset-id", "--motion-id", dest="motion_asset_id", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--rollout-motion-id", dest="motion_id", default=None)
    parser.add_argument("--terrain-id", default=None)
    parser.add_argument("--terrain-urdf", default=None)
    parser.add_argument("--terrain-mesh", default=None)
    parser.add_argument("--surface-catalog", default=None)
    parser.add_argument("--contact-layer", default=None)
    parser.add_argument("--bound-contact-layer", default=None)
    parser.add_argument("--edit-plan", default=None)
    parser.add_argument("--output-contact-layer", default=None)
    parser.add_argument("--raw-contact", action="store_true")
    parser.add_argument("--raw-contact-source", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--source", default="local")

def _cmd_list_motions(_args: argparse.Namespace) -> None:
    ensure_data_dirs()
    for record in list_motion_assets():
        print(
            f"{record.motion_asset_id}\tmotion_id={record.motion_id or record.motion_asset_id}"
            f"\tmotion={record.motion_path}\tterrain={record.terrain_urdf or '-'}"
        )

def _cmd_show_motion(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    record = read_motion_asset(args.motion_id)
    print(record.to_dict())

def _cmd_register_motion_version(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    record = MotionVersionRecord(
        motion_version_id=args.motion_version_id,
        motion_path=str(Path(args.motion).expanduser()),
        kind=args.kind,
        base_motion_id=args.base_motion_id,
        motion_asset_id=args.motion_asset_id,
        parent_motion_version_id=args.parent_motion_version_id,
        contact_layer=args.contact_layer,
        edit_plan_id=args.edit_plan_id,
    )
    out = write_motion_version(record)
    print(f"registered motion version {record.motion_version_id} path={out}")

def _source_segments_for_motion(source: str | None, motion_id: str) -> list:
    if not source:
        return []
    path = LAYERS_ROOT / source / f"{motion_id}.jsonl"
    if not path.exists():
        return []
    default_status = "candidate" if source.startswith("candidates/") else "manual"
    return read_layer(path, default_source=source.split("/", 1)[-1], default_status=default_status)

def _cmd_build_canonical_segmentation(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    motion_path = str(Path(args.motion).expanduser())
    motion_id = args.motion_id or Path(motion_path).stem
    graph = read_contact_graph(LAYERS_ROOT / args.contact_layer, motion_id)
    source_segments = _source_segments_for_motion(args.source, motion_id)
    segments = build_canonical_segments(
        motion_version_id=args.motion_version_id,
        motion_path=motion_path,
        graph=graph,
        source_segments=source_segments,
        cut_source=args.cut_source,
    )
    record, segment_path = write_motion_version_with_canonical_segments(
        motion_version_id=args.motion_version_id,
        motion_path=motion_path,
        contact_layer=args.contact_layer,
        segments=segments,
        base_motion_id=motion_id,
        reset_canonical=args.reset_canonical or args.overwrite,
        reason=args.reason,
        source=args.cut_source,
    )
    print(
        f"wrote canonical segmentation motion_version={record.motion_version_id} "
        f"segments={len(segments)} path={segment_path}"
    )

def _cmd_mark_segment_status(args: argparse.Namespace) -> None:
    updates = []
    if args.status_by_file:
        updates.extend(read_jsonl(Path(args.status_by_file).expanduser()))
    if args.segment_id:
        if not args.status:
            raise ValueError("--segment-id requires --status")
        updates.extend(
            {
                "segment_id": segment_id,
                "status": args.status,
                "reason": args.reason,
                "source": args.source,
            }
            for segment_id in args.segment_id
        )
    if not updates:
        raise ValueError("mark-segment-status requires --segment-id or --status-by-file")
    segments = mark_canonical_segment_statuses(motion_version_id=args.motion_version_id, updates=updates)
    print(f"updated canonical segment status motion_version={args.motion_version_id} updates={len(updates)}")
    print(f"canonical segments={len(segments)}")

def _cmd_canonical_action(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    segments = read_canonical_segments(args.motion_version_id)
    selected = select_segment(
        segments,
        motion_id=args.motion_id,
        segment_id=args.segment_id,
        index=args.index,
    )
    if args.action == "trim":
        if args.start_frame is None or args.end_frame is None:
            raise ValueError("trim requires --start-frame and --end-frame")
        replacements = [trim_segment(selected, start_frame=args.start_frame, end_frame=args.end_frame)]
    elif args.action == "split":
        if args.frame is None:
            raise ValueError("split requires --frame")
        replacements = list(split_segment(selected, frame=args.frame))
    elif args.action == "delete":
        replacements = []
    else:
        raise ValueError(f"unsupported canonical action: {args.action}")
    updated = replace_segment(segments, selected.segment_id, replacements)
    out = replace_canonical_segments(
        args.motion_version_id,
        updated,
        reason=args.reason or f"{args.action} {selected.segment_id}",
        source=args.source,
        kind=args.action,
    )
    print(
        f"{args.action}: motion_version={args.motion_version_id} "
        f"selected={selected.segment_id} replacements={','.join(segment.segment_id for segment in replacements) or '<deleted>'}"
    )
    print(f"updated canonical segmentation {out}")

def _cmd_build_token_catalog(args: argparse.Namespace) -> None:
    segments = read_canonical_segments(args.motion_version_id)
    tokens = build_tokens_from_segments(args.motion_version_id, segments)
    out = write_token_catalog(args.motion_version_id, tokens, args.output)
    try:
        version = read_motion_version(args.motion_version_id)
        metadata = dict(version.metadata)
        metadata["token_catalog_status"] = "current"
        metadata.pop("token_catalog_stale_reason", None)
        updated = replace(version, token_catalog_path=str(out), metadata=metadata)
        write_motion_version(updated)
    except FileNotFoundError:
        print(f"warning: motion version not found for {args.motion_version_id}; token catalog path was not linked")
    print(f"wrote token catalog motion_version={args.motion_version_id} tokens={len(tokens)} path={out}")

def _cmd_summarize(_args: argparse.Namespace) -> None:
    ensure_data_dirs()
    for root in [LAYERS_ROOT / "manual", LAYERS_ROOT / "candidates", LAYERS_ROOT / "accepted", LAYERS_ROOT / "rejected"]:
        files = sorted(root.rglob("*.jsonl"))
        count = 0
        for path in files:
            count += sum(1 for _ in path.open("r", encoding="utf-8"))
        print(f"{root.relative_to(LAYERS_ROOT)}: files={len(files)} segments={count}")

def _cmd_detect_motion(args: argparse.Namespace) -> None:
    paths = detect_omniretarget_paths(args.motion, repo_root=args.repo_root, dataset_root=args.dataset_root)
    print(f"motion_path: {paths.motion_path}")
    print(f"dataset_root: {paths.dataset_root}")
    print(f"terrain_urdf: {paths.terrain_urdf}")
    print(f"terrain_obj: {paths.terrain_obj}")
    print(f"contact_force_npz: {paths.contact_force_npz}")
    print(f"holosoma_motion_path: {paths.holosoma_motion_path}")

def _cmd_export_manifest(args: argparse.Namespace) -> None:
    if getattr(args, "source", None) is not None:
        raise ValueError("Legacy --source export is retired; use --motion-version-id")
    version = read_motion_version(args.motion_version_id)
    segments = read_canonical_segments(args.motion_version_id)
    output = export_motion_version_manifest(args.output, version=version, segments=segments)
    print(f"wrote manifest {output}")

def _cmd_export_split_npz(args: argparse.Namespace) -> None:
    if getattr(args, "source", None) is not None:
        raise ValueError("Legacy --source export is retired; use --motion-version-id")
    version = read_motion_version(args.motion_version_id)
    segments = read_canonical_segments(args.motion_version_id)
    if args.status:
        segments = [segment for segment in segments if segment.status == args.status]
    token_by_segment = {}
    if version.token_catalog_path:
        try:
            token_by_segment = {token.segment_id: token.token_id for token in read_token_catalog(args.motion_version_id, version.token_catalog_path)}
        except FileNotFoundError:
            print(f"warning: token catalog not found: {version.token_catalog_path}")
    hardened_segments = []
    for segment in segments:
        metadata = dict(segment.metadata)
        metadata["motion_version_id"] = args.motion_version_id
        token_id = token_by_segment.get(segment.segment_id)
        if token_id:
            metadata["token_id"] = token_id
        hardened_segments.append(
            replace(
                segment,
                motion_path=segment.motion_path or segment.clip_npz or version.motion_path,
                clip_npz=segment.clip_npz or segment.motion_path or version.motion_path,
                metadata=metadata,
            )
        )
    segments = hardened_segments
    default_output_name = args.motion_version_id if args.status is None else f"{args.motion_version_id}_{args.status}"
    if args.motion_id or args.segment_id or args.index is not None:
        segments = filter_segments(
            segments,
            motion_id=args.motion_id,
            segment_ids=set(args.segment_id or []) if args.segment_id else None,
            indices=set(args.index or []) if args.index is not None else None,
        )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else EXPORTS_ROOT / "split_npz" / default_output_name
    written = export_split_npz(output_dir, segments)
    print(f"wrote {len(written)} split npz files to {output_dir}")

def _cmd_list_contact_layer(args: argparse.Namespace) -> None:
    root = LAYERS_ROOT / args.source
    if not root.exists():
        raise FileNotFoundError(f"contact layer not found: {root}")
    event_files = sorted((root / "events").glob("*.jsonl"))
    if args.motion_id:
        event_files = [path for path in event_files if path.stem == args.motion_id]
    total_events = 0
    total_anchors = 0
    total_patches = 0
    total_transitions = 0
    for event_path in event_files:
        motion_id = event_path.stem
        graph = read_contact_graph(root, motion_id)
        total_events += len(graph.events)
        total_anchors += len(graph.anchors)
        total_patches += len(graph.patches)
        total_transitions += len(graph.transitions)
        print(
            f"{motion_id}: events={len(graph.events)} anchors={len(graph.anchors)} "
            f"patches={len(graph.patches)} transitions={len(graph.transitions)}"
        )
        for anchor in graph.anchors[: args.limit]:
            print(
                f"  anchor {anchor.start_frame:5d}->{anchor.end_frame:<5d} "
                f"{anchor.anchor_id} body={anchor.body} role={anchor.role}"
            )
        for patch in graph.patches[: args.limit]:
            print(
                f"  patch  {patch.start_frame:5d}->{patch.end_frame:<5d} "
                f"{patch.patch_id} body={patch.body} anchor={patch.anchor_id}"
            )
        for transition in graph.transitions[: args.limit]:
            print(
                f"  transition {transition.start_frame:5d}->{transition.end_frame:<5d} "
                f"{transition.transition_id} active={transition.active_body} "
                f"support={','.join(transition.support_bodies)}"
            )
    print(
        f"total_events={total_events} total_anchors={total_anchors} "
        f"total_patches={total_patches} total_transitions={total_transitions}"
    )

def _add_catalog_commands(sub) -> None:
    p = sub.add_parser('init')
    p.set_defaults(func=_cmd_init)
    p = sub.add_parser('import-force-proto')
    p.add_argument('--motion-dir', required=True)
    p.add_argument('--pattern', default='*.npz')
    p.add_argument('--layer-name', default='force_contact')
    p.add_argument('--source', default='force_contact')
    p.add_argument('--motion-id', default=None)
    p.add_argument('--surface-catalog', default=None)
    p.add_argument('--max-surface-distance', type=float, default=0.08)
    p.add_argument('--use-registered-motion-ids', action='store_true')
    p.add_argument('--update-motions', action='store_true')
    p.set_defaults(func=_cmd_import_force_proto)
    p = sub.add_parser('import-asset-manifest')
    p.add_argument('--manifest', required=True)
    p.add_argument('--layer-name', default='newton_8part')
    p.add_argument('--motion-id', action='append', default=None, help='Import one climb_XX or numeric motion id; repeatable')
    p.add_argument('--fps', type=float, default=50.0)
    p.add_argument('--verify-hashes', action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=_cmd_import_asset_manifest)
    p = sub.add_parser('export-contact-overlay')
    p.add_argument('--source', required=True, help='Contact layer path relative to data/layers, e.g. contact/force_contact')
    p.add_argument('--motion-id', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=_cmd_export_contact_overlay)
    p = sub.add_parser('export-surface-binding-report')
    p.add_argument('--contact-layer', required=True, help='Contact layer path relative to data/layers')
    p.add_argument('--motion-id', required=True)
    p.add_argument('--surface-catalog', default=None)
    p.add_argument('--output', required=True)
    p.set_defaults(func=_cmd_export_surface_binding_report)
    p = sub.add_parser('export-surface-binding-overlay')
    p.add_argument('--contact-layer', required=True, help='Contact layer path relative to data/layers')
    p.add_argument('--motion-id', required=True)
    p.add_argument('--surface-catalog', default=None)
    p.add_argument('--output', required=True)
    p.set_defaults(func=_cmd_export_surface_binding_overlay)
    p = sub.add_parser('summarize-surface-bindings')
    p.add_argument('--contact-layer', required=True, help='Contact layer path relative to data/layers')
    p.add_argument('--motion-id', required=True)
    p.add_argument('--surface-catalog', default=None)
    p.add_argument('--limit', type=int, default=10)
    p.set_defaults(func=_cmd_summarize_surface_bindings)
    p = sub.add_parser('bind-contact-surfaces')
    p.add_argument('--contact-layer', default=None, help='Contact layer path relative to data/layers, e.g. contact/force_contact')
    p.add_argument('--motion-id', required=True)
    p.add_argument('--surface-catalog', default=None)
    p.add_argument('--terrain-urdf', default=None)
    p.add_argument('--terrain-mesh', default=None)
    p.add_argument('--include-side-surfaces', action='store_true')
    p.add_argument('--no-ground', action='store_true')
    p.add_argument('--ground-z', type=float, default=0.0)
    p.add_argument('--ground-half-extent', type=float, default=10.0)
    p.add_argument('--output-contact-layer', required=True)
    p.add_argument('--max-distance', type=float, default=0.05)
    p.add_argument('--mode', choices=('reject', 'clamp'), default='reject')
    p.add_argument('--motion-version-id', default=None)
    p.add_argument('--update-motion-version', action='store_true')
    p.add_argument('--rebind-canonical-segments', action='store_true')
    p.set_defaults(func=_cmd_bind_contact_surfaces)
    p = sub.add_parser('create-box-surface-catalog', help='[diagnostic] create a simple box surface catalog')
    p.add_argument('--motion-id', required=True)
    p.add_argument('--box', action='append', required=True, help='object_id:cx,cy,cz:sx,sy,sz')
    p.add_argument('--top-only', action='store_true')
    p.add_argument('--output', required=True)
    p.set_defaults(func=_cmd_create_box_surface_catalog)
    p = sub.add_parser('create-urdf-surface-catalog', help='[internal] create surface catalog from terrain URDF')
    p.add_argument('--motion-id', required=True)
    p.add_argument('--terrain-urdf', required=True)
    p.add_argument('--include-side-surfaces', action='store_true')
    p.add_argument('--no-ground', action='store_true')
    p.add_argument('--ground-z', type=float, default=0.0)
    p.add_argument('--ground-half-extent', type=float, default=10.0)
    p.add_argument('--output', required=True)
    p.set_defaults(func=_cmd_create_urdf_surface_catalog)
    p = sub.add_parser('register-motion-asset')
    p.add_argument('--motion-asset-id', required=True)
    p.add_argument('--motion', required=True)
    p.add_argument('--motion-id', default=None)
    p.add_argument('--terrain-id', default=None)
    p.add_argument('--terrain-urdf', default=None)
    p.add_argument('--terrain-mesh', default=None)
    p.add_argument('--surface-catalog', default=None)
    p.add_argument('--contact-layer', default=None)
    p.add_argument('--bound-contact-layer', default=None)
    p.add_argument('--edit-plan', default=None)
    p.add_argument('--output-contact-layer', default=None)
    p.add_argument('--raw-contact', action='store_true')
    p.add_argument('--raw-contact-source', default=None)
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--source', default='local')
    p.set_defaults(func=_cmd_register_motion_asset)
    p = sub.add_parser('register-motion')
    _add_register_motion_args(p)
    p.set_defaults(func=_cmd_register_motion_asset)
    p = sub.add_parser('list-motions')
    p.set_defaults(func=_cmd_list_motions)
    p = sub.add_parser('show-motion')
    p.add_argument('--motion-id', required=True)
    p.set_defaults(func=_cmd_show_motion)
    p = sub.add_parser('register-motion-version')
    p.add_argument('--motion-version-id', required=True)
    p.add_argument('--motion', required=True)
    p.add_argument('--kind', choices=('raw', 'augmented'), default='raw')
    p.add_argument('--base-motion-id', default=None)
    p.add_argument('--motion-asset-id', default=None)
    p.add_argument('--parent-motion-version-id', default=None)
    p.add_argument('--edit-plan-id', default=None)
    p.add_argument('--contact-layer', default=None)
    p.set_defaults(func=_cmd_register_motion_version)
    p = sub.add_parser('build-canonical-segmentation')
    p.add_argument('--motion-version-id', required=True)
    p.add_argument('--motion', required=True)
    p.add_argument('--motion-id', default=None)
    p.add_argument('--contact-layer', required=True)
    p.add_argument('--source', default=None)
    p.add_argument('--cut-source', default='contact_auto')
    p.add_argument('--reset-canonical', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--reason', default=None)
    p.set_defaults(func=_cmd_build_canonical_segmentation)
    p = sub.add_parser('mark-segment-status', help='[advanced] update canonical segment status')
    p.add_argument('--motion-version-id', required=True)
    p.add_argument('--segment-id', action='append', default=None)
    p.add_argument('--status', choices=('candidate', 'accepted', 'rejected', 'manual'), default=None)
    p.add_argument('--reason', default=None)
    p.add_argument('--source', default='mark_segment_status')
    p.add_argument('--status-by-file', default=None)
    p.set_defaults(func=_cmd_mark_segment_status)
    p = sub.add_parser('canonical-action')
    p.add_argument('--motion-version-id', required=True)
    p.add_argument('--segment-id', default=None)
    p.add_argument('--motion-id', default=None)
    p.add_argument('--index', type=int, default=None)
    p.add_argument('--action', required=True, choices=('trim', 'split', 'delete'))
    p.add_argument('--start-frame', type=int, default=None)
    p.add_argument('--end-frame', type=int, default=None)
    p.add_argument('--frame', type=int, default=None)
    p.add_argument('--reason', default=None)
    p.add_argument('--source', default='canonical_action')
    p.set_defaults(func=_cmd_canonical_action)
    p = sub.add_parser('build-token-catalog')
    p.add_argument('--motion-version-id', required=True)
    p.add_argument('--output', default=None)
    p.set_defaults(func=_cmd_build_token_catalog)
    p = sub.add_parser('export-manifest')
    p.add_argument('--motion-version-id', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=_cmd_export_manifest)
    p = sub.add_parser('export-split-npz')
    p.add_argument('--motion-version-id', required=True)
    p.add_argument('--status', choices=('candidate', 'accepted', 'rejected', 'manual'), default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--motion-id', default=None)
    p.add_argument('--segment-id', action='append', default=None)
    p.add_argument('--index', action='append', type=int, default=None)
    p.set_defaults(func=_cmd_export_split_npz)
    p = sub.add_parser('list-contact-layer')
    p.add_argument('--source', required=True, help='Contact layer path relative to data/layers, e.g. contact/force_contact')
    p.add_argument('--motion-id', default=None)
    p.add_argument('--limit', type=int, default=5)
    p.set_defaults(func=_cmd_list_contact_layer)
    p = sub.add_parser('detect-motion', help='[diagnostic] inspect inferred terrain/contact paths for a motion')
    p.add_argument('motion')
    p.add_argument('--repo-root', default=None)
    p.add_argument('--dataset-root', default=None)
    p.set_defaults(func=_cmd_detect_motion)
    p = sub.add_parser('summarize')
    p.set_defaults(func=_cmd_summarize)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage motion catalogs, inspect recorded data, and export canonical segments.")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_catalog_commands(sub)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
