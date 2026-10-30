"""Copy evidence without changing its original definitions or source files."""

import hashlib
import html
import json
import shutil
from pathlib import Path

EXPERIMENTS = (
    "predictor_establish_visibility_20261013",
    "legacy_camera_route_coverage_20261013",
    "predictor_wide_depth_20261013",
    "predictor_depth_scope_sweep_20261013",
    "predictor_yaw_depth_20261013",
    "predictor_root_depth_20261013",
)
REPORTS = (
    "report.json",
    "adjusted_report.json",
    "pitch85_report.json",
    "original_size_report.json",
    "selected_report.json",
    "final_report.json",
)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def import_history(source_root, output):
    source_root = source_root.resolve()
    output = output.resolve()
    if output == source_root or source_root.is_relative_to(output):
        raise ValueError("The output must not contain or overwrite the historical source root")
    sources = [source_root / name for name in EXPERIMENTS]
    for source in sources:
        if not (source / "report.json").is_file():
            raise FileNotFoundError(source / "report.json")
    route = read_json(sources[1] / "report.json")
    scene_metadata = Path(route["metadata_source"])
    if not scene_metadata.is_file():
        raise FileNotFoundError(scene_metadata)
    jobs = []
    for source in sources:
        destination = output / "history" / source.name
        for path in sorted(source.iterdir()):
            if path.is_file():
                jobs.append((path, destination / path.name))
    jobs.append((scene_metadata, output / "history" / sources[1].name / "scene_metadata.json"))
    for name in ("predictor_root_depth_run_20261013.py", "predictor_yaw_depth_run_20261013.py"):
        source = source_root / name
        if source.is_file():
            jobs.append((source, output / "history" / "source_scripts" / name))
    # Check every destination before copying: reruns are idempotent and never
    # silently overwrite modified evidence or write through a source symlink.
    for source, destination in jobs:
        if destination.resolve() == source.resolve():
            raise ValueError(f"Source and destination are identical: {source}")
        if not destination.resolve().is_relative_to(output):
            raise ValueError(f"Destination escapes the import directory: {destination}")
        if destination.exists() and digest(destination) != digest(source):
            raise FileExistsError(f"Evidence differs; choose a fresh --output: {destination}")
    files = []
    for source, destination in jobs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        sha256 = digest(source)
        if digest(destination) != sha256:
            raise ValueError(f"Copy verification failed: {destination}")
        files.append(
            dict(
                source=str(source),
                file=str(destination.relative_to(output)),
                bytes=destination.stat().st_size,
                sha256=sha256,
            )
        )
    manifest = dict(
        schema="contact_observability_history_v1",
        files=files,
        source_root=str(source_root),
        source_files_preserved=True,
        historical_contact_truth_revalidated=False,
        scope="Historical geometry/visibility diagnostics; not newly materialized contact labels",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    links = []
    for name in EXPERIMENTS:
        base = "history/" + name
        links.append(
            f'<li><a href="{base}/report.json">{html.escape(name)} · report</a>'
            + (f' · <a href="{base}/index.html">交互页面</a>' if (output / base / "index.html").exists() else "")
            + (f' · <a href="{base}/targets.csv">目标 CSV</a>' if (output / base / "targets.csv").exists() else "")
            + "</li>"
        )
    run_links = []
    for run in route_runs(output):
        name = html.escape(run["directory"], quote=True)
        run_links.append(
            f'<li><a href="{name}/report.json">{name} · 报告</a>'
            + (
                f' · <a href="{name}/index.html">查看器</a>'
                if (output / run["directory"] / "index.html").is_file()
                else ""
            )
            + "</li>"
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Contact observability</title>'
        "<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;line-height:1.8}li{margin:16px 0}</style>"
        "<h1>目标接触位置的观测覆盖</h1><p>起始帧观测、历史累计观测、接触前覆盖与全程最终覆盖分别审计。"
        "这些是理想深度几何诊断；不含机器人自遮挡或噪声，不定义实际接触。</p>"
        '<p><a href="summary.json">统一摘要</a> · <a href="manifest.json">来源与 SHA256</a></p>'
        "<ul>" + "".join(links) + "</ul><h2>独立入口复跑</h2><ul>" + "".join(run_links) + "</ul></html>\n"
    )


def route_runs(output):
    runs = []
    for directory in sorted(output.iterdir()):
        path = directory / "report.json"
        if not directory.is_dir() or not path.is_file():
            continue
        report = read_json(path)
        if report.get("schema") != "legacy_depth_route_sampled_surface_coverage_v1":
            continue
        runs.append(
            dict(
                directory=directory.name,
                motion=report["motion"]["motion_id"],
                frames=report["motion_frames"],
                camera=report["camera"],
                coverage_distance_m=report["coverage_distance_m"],
                metrics=report["target_summary"],
                excluded_targets=len(report["excluded_targets"]),
                contact_truth=False,
            )
        )
    return runs


def summarize(output):
    history = output / "history"
    initial = read_json(history / EXPERIMENTS[0] / "report.json")
    route = read_json(history / EXPERIMENTS[1] / "report.json")
    distance = 4.0  # Historical route definition in cm, not solver contact semantics.
    targets = route["targets"]
    current = sum(row["nearest_at_start_cm"] is not None and row["nearest_at_start_cm"] <= distance for row in targets)
    manifest = read_json(output / "manifest.json")
    comparisons = []
    for name in EXPERIMENTS[2:]:
        for filename in REPORTS:
            path = history / name / filename
            if path.exists():
                report = read_json(path)
                profiles = [
                    {k: p[k] for k in ("fov", "resolution", "pitch", "summary") if k in p}
                    for p in report.get("profiles", [])
                ]
                comparisons.append(
                    dict(
                        experiment=name,
                        report=filename,
                        profiles=profiles,
                        metrics={
                            k: report[k]
                            for k in (
                                "target_count",
                                "old_visible",
                                "new_in_frustum",
                                "actual_scan_candidate_covered",
                                "within_4cm_diagnostic",
                                "nearest_error_cm",
                                "profile",
                                "summary",
                            )
                            if k in report
                        },
                    )
                )
    return dict(
        schema="contact_observability_summary_v1",
        source_files=len(manifest["files"]),
        source_bytes=sum(f["bytes"] for f in manifest["files"]),
        start_frame=dict(
            motions=initial["motions"],
            samples=initial["samples"],
            definition=initial["definition"],
            visibility=initial["visibility"],
            metrics=initial["summary"],
        ),
        full_route=dict(
            motion=route["motion"]["motion_id"],
            frames=route["motion_frames"],
            fps=route["fps"],
            camera=route["camera"],
            coverage_distance_m=distance / 100,
            metrics=dict(
                total=len(targets),
                covered_at_start=current,
                **{k: route["target_summary"][k] for k in ("seen_by_start", "seen_before_target", "seen_anytime")},
            ),
            late_only_targets=[r for r in targets if r["seen_anytime"] and not r["seen_before_target"]],
            area_method=route["area_method"],
            surfaces=route["surfaces"],
        ),
        camera_comparisons=comparisons,
        route_runs=route_runs(output),
        contact_truth=False,
        historical_contact_truth_revalidated=False,
        sensor_model=route["sensor_model"],
        timing="Prefix includes the planning start frame; before-target excludes the touchdown frame; anytime includes later frames",
        definitions="Frustum membership, first-hit visibility, same-surface candidate coverage and sampled-area coverage retain their separate original definitions",
    )
