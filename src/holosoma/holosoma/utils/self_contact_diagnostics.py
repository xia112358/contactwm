from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _robot_body(label: object) -> tuple[str, str] | None:
    text = str(label)
    marker = "/Robot/"
    if marker not in text:
        return None
    robot_path, _, body_path = text.partition(marker)
    return robot_path, body_path.rsplit("/", 1)[-1]


class SelfContactDiagnostics:
    """Accumulate actual same-robot Newton contacts for short diagnostic runs."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.sample_count = 0
        self.raw_contact_count = 0
        self.robot_external_contact_count = 0
        self.unknown_body_contact_count = 0
        self.unknown_examples: list[dict[str, object]] = []
        self.frames_with_self_contact = 0
        self.max_contacts_per_sample = 0
        self._pairs: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"active_samples": 0.0, "contact_count": 0.0, "force_sum_n": 0.0, "force_max_n": 0.0}
        )

    def update(self, raw_contacts: dict[str, np.ndarray], body_labels: list[object]) -> None:
        count = int(np.asarray(raw_contacts["count"]).item())
        body0 = np.asarray(raw_contacts["body0"][:count], dtype=np.int64)
        body1 = np.asarray(raw_contacts["body1"][:count], dtype=np.int64)
        forces = np.linalg.norm(np.asarray(raw_contacts["force_w"][:count]), axis=1)
        active_pairs: set[tuple[str, str]] = set()
        self_contacts = 0
        self.raw_contact_count += count

        shape0 = np.asarray(raw_contacts.get("shape0", np.full(count, -1))[:count], dtype=np.int64)
        shape1 = np.asarray(raw_contacts.get("shape1", np.full(count, -1))[:count], dtype=np.int64)
        for b0, b1, s0, s1, force_n in zip(body0, body1, shape0, shape1, forces):
            # Multiple collision shapes attached to one rigid body are not a
            # robot self-contact pair.
            if int(b0) == int(b1):
                continue
            first = _robot_body(body_labels[int(b0)]) if 0 <= b0 < len(body_labels) else None
            second = _robot_body(body_labels[int(b1)]) if 0 <= b1 < len(body_labels) else None
            if first is None and second is None:
                self.unknown_body_contact_count += 1
                if len(self.unknown_examples) < 10:
                    example = {"body0": int(b0), "body1": int(b1), "shape0": int(s0), "shape1": int(s1)}
                    if 0 <= b0 < len(body_labels):
                        example["label0"] = str(body_labels[int(b0)])
                    if 0 <= b1 < len(body_labels):
                        example["label1"] = str(body_labels[int(b1)])
                    self.unknown_examples.append(example)
                continue
            if first is None or second is None or first[0] != second[0]:
                self.robot_external_contact_count += 1
                continue
            pair = tuple(sorted((first[1], second[1])))
            stats = self._pairs[pair]
            stats["contact_count"] += 1.0
            stats["force_sum_n"] += float(force_n)
            stats["force_max_n"] = max(stats["force_max_n"], float(force_n))
            active_pairs.add(pair)
            self_contacts += 1

        for pair in active_pairs:
            self._pairs[pair]["active_samples"] += 1.0
        self.sample_count += 1
        self.frames_with_self_contact += int(self_contacts > 0)
        self.max_contacts_per_sample = max(self.max_contacts_per_sample, self_contacts)

    def write(self) -> None:
        rows = []
        for pair, stats in self._pairs.items():
            contact_count = int(stats["contact_count"])
            rows.append(
                {
                    "body_a": pair[0],
                    "body_b": pair[1],
                    "active_samples": int(stats["active_samples"]),
                    "contact_count": contact_count,
                    "mean_force_n": stats["force_sum_n"] / max(contact_count, 1),
                    "max_force_n": stats["force_max_n"],
                }
            )
        rows.sort(key=lambda row: (-row["active_samples"], -row["contact_count"], row["body_a"], row["body_b"]))
        payload = {
            "schema": "holosoma_newton_self_contact_diagnostics_v1",
            "sample_count": self.sample_count,
            "raw_contact_count": self.raw_contact_count,
            "robot_external_contact_count": self.robot_external_contact_count,
            "unknown_body_contact_count": self.unknown_body_contact_count,
            "unknown_examples": self.unknown_examples,
            "frames_with_self_contact": self.frames_with_self_contact,
            "max_contacts_per_sample": self.max_contacts_per_sample,
            "pairs": rows,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2) + "\n")
