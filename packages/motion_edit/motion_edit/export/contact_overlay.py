from __future__ import annotations

import json
from pathlib import Path

from motion_edit.contact.graph import ContactGraph


def export_contact_overlay(path: str | Path, graph: ContactGraph) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": 1, "contact_graph": graph.to_dict()}, indent=2), encoding="utf-8")
    return out

