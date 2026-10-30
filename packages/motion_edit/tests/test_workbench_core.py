from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from motion_edit.contact import contact_graph_from_masks, write_contact_layer
from motion_edit.io import read_jsonl, write_jsonl
from motion_edit.schema import SegmentRecord
from motion_edit.workbench import (
    WorkbenchSession,
    curate_segment,
    export_cutter_session_file,
    load_workbench_segments,
    replace_segment,
    select_segment,
    split_segment,
    sync_cutter_session_file,
    trim_segment,
    upsert_workbench_segments,
    validate_workbench_segments,
    write_workbench_segments,
)


def _segment(index: int = 0) -> SegmentRecord:
    start, end = ((10, 30), (30, 45))[index]
    return SegmentRecord(
        motion_id="motion_a",
        segment_id=f"motion_a_force_{index:04d}",
        start_frame=start,
        end_frame=end,
        source="force_contact",
        status="candidate",
        motion_path="motion_a.npz",
        clip_npz="motion_a.npz",
    )


def test_workbench_actions_preserve_identity_and_provenance() -> None:
    trimmed = trim_segment(_segment(), start_frame=12, end_frame=28)
    left, right = split_segment(trimmed, frame=18)
    accepted = curate_segment(left, status="accepted")

    assert trimmed.segment_id == "motion_a_force_0000"
    assert trimmed.metadata["motion_edit_edits"][-1]["kind"] == "trim"
    assert (left.start_frame, left.end_frame) == (12, 18)
    assert (right.start_frame, right.end_frame) == (18, 28)
    assert accepted.status == "accepted"
    assert accepted.metadata["motion_edit_edits"][-1]["kind"] == "curate"


def test_trim_rejects_implicit_extension() -> None:
    with pytest.raises(ValueError):
        trim_segment(_segment(), start_frame=9, end_frame=28)
    extended = trim_segment(_segment(), start_frame=9, end_frame=31, allow_extend=True)
    assert (extended.start_frame, extended.end_frame) == (9, 31)


def test_workbench_layer_replace_and_upsert(tmp_path: Path) -> None:
    layers_root = tmp_path / "layers"
    write_workbench_segments("candidates/source", [_segment(), _segment(1)], layers_root=layers_root)
    loaded = load_workbench_segments("candidates/source", layers_root=layers_root)
    selected = select_segment(loaded, motion_id="motion_a", index=0)
    updated_first = trim_segment(selected, start_frame=12, end_frame=28)
    replaced = replace_segment(loaded, selected.segment_id, [updated_first])
    write_workbench_segments("manual/workbench_tmp", replaced, layers_root=layers_root)

    upsert_workbench_segments("accepted/probe", [curate_segment(updated_first, status="accepted")], layers_root=layers_root)
    upsert_workbench_segments("accepted/probe", [curate_segment(_segment(1), status="accepted")], layers_root=layers_root)
    accepted = load_workbench_segments("accepted/probe", layers_root=layers_root)

    assert [(item.start_frame, item.end_frame) for item in replaced] == [(12, 28), (30, 45)]
    assert [item.segment_id for item in accepted] == ["motion_a_force_0000", "motion_a_force_0001"]


def test_cutter_session_uses_generic_ui_metadata(tmp_path: Path) -> None:
    layers_root = tmp_path / "layers"
    workbench_root = tmp_path / "workbench"
    write_workbench_segments("candidates/source", [_segment(), _segment(1)], layers_root=layers_root)
    graph = contact_graph_from_masks(
        motion_id="motion_a",
        contact_mask=np.asarray([[True, False], [False, False], [True, False]]),
        body_names=["LF", "RF"],
        source="source",
    )
    write_contact_layer(layers_root / "contact" / "source", graph)

    session = export_cutter_session_file(
        motion_id="motion_a",
        source_layer="candidates/source",
        session_name="session_a",
        motion_path="motion_a.npz",
        ui_port=8123,
        layers_root=layers_root,
        workbench_root=workbench_root,
    )
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    records = read_jsonl(session.segment_path)

    assert manifest["ui_port"] == 8123
    assert "viewer_port" not in manifest
    assert manifest["sync_status"] == "prepared"
    assert records[0]["metadata"]["motion_edit_cutter_session"]["original_segment_id"] == "motion_a_force_0000"


def test_cutter_session_sync_uses_workbench_source(tmp_path: Path) -> None:
    layers_root = tmp_path / "layers"
    workbench_root = tmp_path / "workbench"
    write_workbench_segments("candidates/source", [_segment(), _segment(1)], layers_root=layers_root)
    contact = np.zeros((60, 2), dtype=bool)
    contact[:, 1] = True
    contact[12:28, 0] = True
    graph = contact_graph_from_masks(
        motion_id="motion_a",
        contact_mask=contact,
        body_names=["LF", "RF"],
        source="source",
    )
    write_contact_layer(layers_root / "contact" / "source", graph)
    session = export_cutter_session_file(
        motion_id="motion_a",
        source_layer="candidates/source",
        session_name="session_a",
        layers_root=layers_root,
        workbench_root=workbench_root,
    )
    records = read_jsonl(session.segment_path)
    records[0]["start_frame"] = 12
    records[0]["end_frame"] = 28
    write_jsonl(session.segment_path, records)

    sync_cutter_session_file(
        session.segment_path,
        source_layer="candidates/source",
        session_name="session_a",
        layers_root=layers_root,
    )
    synced = load_workbench_segments("manual/session_a", layers_root=layers_root)

    assert all(item.source == "motion_edit_workbench" for item in synced)
    edit = synced[0].metadata["motion_edit_edits"][-1]
    assert edit["kind"] == "import_from_cutter"
    assert edit["source"] == "motion_edit_workbench"


def test_workbench_validation_and_session_accept(tmp_path: Path) -> None:
    duplicate = SegmentRecord(
        motion_id="motion_a",
        segment_id="motion_a_force_0000",
        start_frame=20,
        end_frame=40,
        source="force_contact",
        status="accepted",
    )
    warnings = validate_workbench_segments(
        [curate_segment(_segment(), status="accepted"), duplicate],
        destination="accepted/probe",
    )
    assert any("duplicate segment_id" in warning for warning in warnings)
    assert any("accepted overlap" in warning for warning in warnings)

    layers_root = tmp_path / "layers"
    write_workbench_segments("candidates/source", [_segment(), _segment(1)], layers_root=layers_root)
    session = WorkbenchSession(
        source="candidates/source",
        motion_id="motion_a",
        selected_index=0,
        layer_name="session_probe",
        layers_root=layers_root,
    )
    session.accept()
    session.set_selection(motion_id="motion_a", index=1)
    session.accept()
    accepted = load_workbench_segments("accepted/session_probe", layers_root=layers_root)
    assert [item.segment_id for item in accepted] == ["motion_a_force_0000", "motion_a_force_0001"]
