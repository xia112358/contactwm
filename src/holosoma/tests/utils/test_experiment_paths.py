from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from holosoma.utils.experiment_paths import get_eval_log_dir


def test_eval_log_dir_is_sibling_of_resolved_training_logs(tmp_path: Path) -> None:
    training_logs = tmp_path / "runtime/current/holosoma/logs"
    training_logs.mkdir(parents=True)
    logs_link = tmp_path / "logs"
    logs_link.symlink_to(training_logs, target_is_directory=True)

    path = get_eval_log_dir(
        SimpleNamespace(base_dir=str(logs_link)),
        SimpleNamespace(project="WholeBodyTracking"),
        "20260712_120000",
    )

    assert path == training_logs.parent / "eval/WholeBodyTracking/20260712_120000"
    assert training_logs not in path.parents
