from __future__ import annotations

import argparse
from pathlib import Path

from motion_edit.segmentation.session import (
    add_draft_segment,
    create_segmentation_edit_session,
    delete_draft_segment,
    discard_segmentation_edit_session,
    list_draft_segments,
    read_segmentation_edit_session,
    relabel_draft_segment,
    save_segmentation_edit_session,
    trim_draft_segment,
    validate_segmentation_segments,
)


def _selection_kwargs(args: argparse.Namespace) -> dict:
    return {
        "segment_id": getattr(args, "segment_id", None),
        "motion_id": getattr(args, "motion_id", None),
        "index": getattr(args, "index", None),
    }


def _support_bodies(args: argparse.Namespace) -> list[str]:
    return list(getattr(args, "support_body", None) or [])


def _cmd_start(args: argparse.Namespace) -> None:
    session = create_segmentation_edit_session(
        args.motion_version_id,
        session_id=args.session_id,
        overwrite=args.overwrite,
    )
    segments = list_draft_segments(session)
    print(f"started segmentation edit session {session.session_id}")
    print(f"motion_version_id={session.motion_version_id}")
    print(f"session={Path(session.draft_segment_path).parent / 'session.json'}")
    print(f"draft_segments={session.draft_segment_path}")
    print(f"segments={len(segments)}")


def _cmd_list(args: argparse.Namespace) -> None:
    session = read_segmentation_edit_session(args.session)
    segments = list_draft_segments(session, motion_id=args.motion_id)
    warnings = validate_segmentation_segments(segments, allow_overlap=args.allow_overlap)
    print(f"session={session.session_id} state={session.state} motion_version={session.motion_version_id}")
    print(f"segments={len(segments)}")
    for index, segment in enumerate(segments[: args.limit]):
        metadata = segment.metadata or {}
        review = metadata.get("review") or {}
        active_body = metadata.get("active_body") or segment.active or "-"
        support_bodies = metadata.get("support_bodies") or segment.support or "-"
        transition_type = metadata.get("transition_type") or "-"
        print(
            f"{index:03d} {segment.start_frame:5d}->{segment.end_frame:<5d} "
            f"len={segment.end_frame - segment.start_frame:<4d} status={segment.status:<9} "
            f"review={review.get('state', '-'):<9} active={active_body} "
            f"support={support_bodies} transition={transition_type} id={segment.segment_id}"
        )
    if len(segments) > args.limit:
        print(f"... {len(segments) - args.limit} more")
    for warning in warnings:
        print(f"warning: {warning}")


def _cmd_trim(args: argparse.Namespace) -> None:
    segment = trim_draft_segment(
        args.session,
        **_selection_kwargs(args),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        reason=args.reason,
        allow_overlap=args.allow_overlap,
    )
    print(
        f"trimmed {segment.segment_id}: "
        f"{segment.start_frame}->{segment.end_frame} len={segment.end_frame - segment.start_frame}"
    )


def _cmd_add(args: argparse.Namespace) -> None:
    segment = add_draft_segment(
        args.session,
        motion_id=args.motion_id,
        segment_id=args.segment_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        active=args.active,
        support=args.support,
        active_body=args.active_body,
        support_bodies=_support_bodies(args),
        transition_type=args.transition_type,
        status=args.status,
        reason=args.reason,
        allow_overlap=args.allow_overlap,
    )
    print(
        f"added {segment.segment_id}: "
        f"{segment.start_frame}->{segment.end_frame} status={segment.status}"
    )


def _cmd_delete(args: argparse.Namespace) -> None:
    segment = delete_draft_segment(args.session, **_selection_kwargs(args), reason=args.reason)
    print(f"deleted {segment.segment_id}: {segment.start_frame}->{segment.end_frame}")


def _cmd_relabel(args: argparse.Namespace) -> None:
    segment = relabel_draft_segment(
        args.session,
        **_selection_kwargs(args),
        active=args.active,
        support=args.support,
        active_body=args.active_body,
        support_bodies=_support_bodies(args),
        transition_type=args.transition_type,
        status=args.status,
        reason=args.reason,
    )
    metadata = segment.metadata or {}
    print(
        f"relabeled {segment.segment_id}: status={segment.status} "
        f"active={metadata.get('active_body') or segment.active} "
        f"support={metadata.get('support_bodies') or segment.support} "
        f"transition={metadata.get('transition_type') or '-'}"
    )


def _cmd_save(args: argparse.Namespace) -> None:
    out = save_segmentation_edit_session(args.session, reason=args.reason, allow_overlap=args.allow_overlap)
    print(f"saved segmentation edit session to canonical segments: {out}")


def _cmd_discard(args: argparse.Namespace) -> None:
    session = discard_segmentation_edit_session(args.session, reason=args.reason)
    print(f"discarded segmentation edit session {session.session_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motion-edit-seg",
        description="Explicit segmentation edit sessions: start from automatic canonical cuts, edit a draft with wrapper-owned timeline handles, then save or discard.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start", help="start a draft segmentation edit session from canonical segments")
    p.add_argument("--motion-version-id", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_start)

    p = sub.add_parser("list", help="list draft segments in an edit session")
    p.add_argument("--session", required=True)
    p.add_argument("--motion-id", default=None)
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--allow-overlap", action="store_true", help="print overlap warnings instead of failing validation")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("trim", help="edit the selected segment boundary in the draft")
    _add_selection_args(p)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--allow-overlap", action="store_true")
    p.add_argument("--reason", default=None)
    p.set_defaults(func=_cmd_trim)

    p = sub.add_parser("add", help="add a manual segment to the draft")
    p.add_argument("--session", required=True)
    p.add_argument("--motion-id", default=None)
    p.add_argument("--segment-id", default=None)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--active", default=None)
    p.add_argument("--support", default=None)
    p.add_argument("--active-body", default=None)
    p.add_argument("--support-body", action="append", default=None)
    p.add_argument("--transition-type", default=None)
    p.add_argument("--status", choices=("candidate", "accepted", "rejected", "manual"), default="manual")
    p.add_argument("--allow-overlap", action="store_true")
    p.add_argument("--reason", default=None)
    p.set_defaults(func=_cmd_add)

    p = sub.add_parser("delete", help="delete the selected segment from the draft")
    _add_selection_args(p)
    p.add_argument("--reason", default=None)
    p.set_defaults(func=_cmd_delete)

    p = sub.add_parser("relabel", help="relabel the selected draft segment")
    _add_selection_args(p)
    p.add_argument("--active", default=None)
    p.add_argument("--support", default=None)
    p.add_argument("--active-body", default=None)
    p.add_argument("--support-body", action="append", default=None)
    p.add_argument("--transition-type", default=None)
    p.add_argument("--status", choices=("candidate", "accepted", "rejected", "manual"), default=None)
    p.add_argument("--reason", default=None)
    p.set_defaults(func=_cmd_relabel)

    p = sub.add_parser("save", help="replace canonical segmentation with the session draft")
    p.add_argument("--session", required=True)
    p.add_argument("--allow-overlap", action="store_true")
    p.add_argument("--reason", default=None)
    p.set_defaults(func=_cmd_save)

    p = sub.add_parser("discard", help="close a draft session without touching canonical segments")
    p.add_argument("--session", required=True)
    p.add_argument("--reason", default=None)
    p.set_defaults(func=_cmd_discard)

    return parser


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", required=True)
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--motion-id", default=None)
    parser.add_argument("--index", type=int, default=None)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
