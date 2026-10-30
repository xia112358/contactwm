"""Explicit local path aliases for immutable published training artifacts.

The installed mapping is local metadata, not an amendment to source evidence.
Only declared absolute path strings are relocated; hashes, relative references,
dictionary keys, tensors and contact/asset semantics remain unchanged.
The installer updates mappings by atomic replacement; parsed revisions are
cached by file identity and metadata rather than rereading each path lookup.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA = "somaforge_published_data_paths_v1"
_EMPTY_RULES = (MappingProxyType({}), ())


def _mapping_path() -> Path:
    root = Path(os.environ.get("SOMAFORGE_ROOT", Path(__file__).resolve().parents[3]))
    return root / "runtime/current/manifests/published_data_paths.json"


def _absolute_path(value: Any, *, context: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or not Path(value).is_absolute():
        raise ValueError(f"{context} must be a nonempty absolute path")
    # Normalize lexically: resolving old symlinks would change the source alias
    # before it can match its declared original spelling.
    return Path(os.path.normpath(value))


@lru_cache(maxsize=4)
def _read_rules(path: Path, _stamp: tuple[int, ...]) -> tuple[Mapping[Path, Path], tuple[tuple[Path, Path], ...]]:
    """Cache only immutable parsed rules for one specific on-disk revision."""
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported published-data path mapping: {path}")
    rules: dict[str, dict[Path, Path]] = {}
    for kind in ("files", "prefixes"):
        rows = data.get(kind, [])
        if not isinstance(rows, list):
            raise ValueError(f"Published-data {kind} must be a list: {path}")
        parsed: dict[Path, Path] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"source", "target"}:
                raise ValueError(f"Published-data {kind} entries require source and target: {path}")
            source = _absolute_path(row["source"], context=f"{kind}.source")
            target = _absolute_path(row["target"], context=f"{kind}.target")
            if source in parsed and parsed[source] != target:
                raise ValueError(f"Conflicting published-data {kind} mapping for {source}")
            parsed[source] = target
        rules[kind] = parsed
    prefixes = tuple(sorted(rules["prefixes"].items(), key=lambda pair: len(pair[0].parts), reverse=True))
    return MappingProxyType(rules["files"]), prefixes


def _load_rules() -> tuple[Mapping[Path, Path], tuple[tuple[Path, Path], ...]]:
    path = _mapping_path().resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        # Never retain an absent file: a later install must take effect on the
        # next read, as must moving an existing mapping out of the way.
        return _EMPTY_RULES
    # Inode/device catch an atomic replacement even when size and mtime were
    # preserved. ctime catches in-place rewrites when the filesystem reports
    # changed metadata, including a restored mtime.
    stamp = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    return _read_rules(path, stamp)


def _relocated_path(value: str | os.PathLike[str], rules) -> Path | None:
    path = Path(value)
    if not path.is_absolute():
        return None
    path = Path(os.path.normpath(os.fspath(path)))
    files, prefixes = rules
    if path in files:
        return files[path]
    for source, target in prefixes:
        if path.is_relative_to(source):
            # An installed checkout may be nested inside the historical data
            # root (for example an isolated release validation checkout). Do
            # not reinterpret its already relocated paths as old inputs.
            if any(len(installed.parts) > len(source.parts)
                   and path.is_relative_to(installed) for _, installed in prefixes):
                return path
            return target / path.relative_to(source)
    return None


def _resolve(value: str | os.PathLike[str], base_dir, rules) -> Path:
    path = Path(value).expanduser()
    relocated = _relocated_path(path, rules)
    if relocated is not None:
        path = relocated
    elif not path.is_absolute() and base_dir is not None:
        base = _resolve(base_dir, None, rules)
        # A published relative reference can cross from an old physical data
        # tree into its sibling checkout. The new checkout/data layout need
        # not retain that relationship. Reconstruct the original base using
        # declared aliases only, without resolving any old-machine symlinks.
        targets = ((source, target.resolve()) for source, target in rules[1])
        for source, target in sorted(targets, key=lambda pair: len(pair[1].parts), reverse=True):
            if base.is_relative_to(target):
                original_base = source / base.relative_to(target)
                original_candidate = Path(os.path.normpath(os.fspath(original_base / path)))
                restored = _relocated_path(original_candidate, rules)
                if restored is not None:
                    return restored.resolve()
                break
        path = base / path
    return path.resolve()


def resolve_published_path(value: str | os.PathLike[str], base_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a file using explicit aliases, or ordinary pathlib resolution.

    Exact file mappings take priority over directory prefixes. Prefixes match
    whole path components, most specific first. A mapping is applied once;
    targets are never recursively remapped.
    Relative references under an installed target retain their declared source
    base when crossing to another explicitly mapped source tree.
    """
    return _resolve(value, base_dir, _load_rules())


def _relocate_values(value: Any, rules) -> Any:
    if isinstance(value, str):
        path = _relocated_path(value, rules)
        return str(path) if path is not None else value
    if isinstance(value, dict):
        return {key: _relocate_values(item, rules) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate_values(item, rules) for item in value]
    if isinstance(value, tuple):
        return tuple(_relocate_values(item, rules) for item in value)
    return value


def relocate_published_values(value: Any) -> Any:
    """Relocate declared absolute strings in metadata without mutating it.

    Dictionary keys and non-container objects are deliberately left alone.
    Relative references retain their source document's existing interpretation.
    """
    return _relocate_values(value, _load_rules())


def load_published_json(path: str | os.PathLike[str]) -> Any:
    """Read an aliased JSON path and relocate its metadata in memory only."""
    rules = _load_rules()
    with _resolve(path, None, rules).open(encoding="utf-8") as stream:
        return _relocate_values(json.load(stream), rules)


def load_published_tensor_input(path):
    """Relocate input metadata without traversing model/optimizer/record tensors."""
    import torch
    loaded = torch.load(resolve_published_path(path), map_location='cpu', weights_only=False)
    result = dict(loaded)
    for key in ('baseline', 'predictor', 'corpus', 'parent_corpus', 'material_source_corpus',
                'source_audit', 'refinement', 'initialization', 'metadata', 'extension', 'config'):
        if key in loaded:
            result[key] = relocate_published_values(loaded[key])
    if isinstance(result.get('config'), dict):
        # torch-saved dataclass configurations can contain actual Path objects.
        result['config'] = {key: resolve_published_path(value) if isinstance(value, Path) else value
                            for key, value in result['config'].items()}
    return result
