import pytest

from somaforge_core.newton_contact_data import (
    require_current_newton_manifest,
    require_current_newton_provenance,
)
from somaforge_core.newton_runtime_compat import NEWTON_RUNTIME_ID


def test_current_newton_contact_provenance_is_accepted() -> None:
    semantics = {
        "provenance": {
            "newton_runtime": {"runtime_id": NEWTON_RUNTIME_ID},
        }
    }
    runtime = require_current_newton_provenance(semantics, context="test")
    assert runtime["runtime_id"] == NEWTON_RUNTIME_ID


@pytest.mark.parametrize("runtime", [None, {}, {"runtime_id": "newton-1.2.0"}])
def test_old_or_unrecorded_newton_contact_provenance_is_rejected(runtime) -> None:
    semantics = {"provenance": {}}
    if runtime is not None:
        semantics["provenance"]["newton_runtime"] = runtime
    with pytest.raises(ValueError, match="Relabel the unchanged motion"):
        require_current_newton_provenance(semantics, context="test")


def test_dataset_manifest_must_declare_current_newton_runtime() -> None:
    with pytest.raises(ValueError, match="Rebuild labels, events, and caches together"):
        require_current_newton_manifest({}, context="old manifest")
    runtime = require_current_newton_manifest(
        {"newton_runtime": {"runtime_id": NEWTON_RUNTIME_ID}}, context="new manifest"
    )
    assert runtime["runtime_id"] == NEWTON_RUNTIME_ID
