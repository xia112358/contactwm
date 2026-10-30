"""Manifest-first registry for all SomaForge runtime assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ASSET_MANIFEST_SCHEMA = "somaforge_asset_manifest_v1"


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    kind: str
    status: str
    files: Mapping[str, str]
    sha256: Mapping[str, str]
    metadata: Mapping[str, Any]


class AssetManifest:
    def __init__(self, path: str | Path, payload: Mapping[str, Any]) -> None:
        self.path = Path(path).expanduser().resolve()
        if payload.get("schema") != ASSET_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported asset manifest schema: {payload.get('schema')!r}")
        self.root = (self.path.parent / str(payload.get("root", ".."))).resolve()
        external_roots = payload.get("external_roots", [])
        if not isinstance(external_roots, list):
            raise ValueError("external_roots must be a list of relative directory paths")
        roots = {}
        for value in external_roots:
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid external root: {value!r}")
            alias = Path(value)
            if alias.is_absolute() or alias == Path(".") or ".." in alias.parts:
                raise ValueError(f"external root must be a relative directory without traversal: {value!r}")
            if alias in roots:
                raise ValueError(f"duplicate external root: {value!r}")
            roots[alias] = (self.root / alias).resolve()
        self.external_roots = tuple(roots.items())
        records = payload.get("assets")
        if not isinstance(records, list):
            raise ValueError(f"asset manifest has no assets list: {self.path}")
        self._records = {}
        for raw in records:
            if not isinstance(raw, dict):
                raise ValueError(f"asset manifest entry must be an object: {raw!r}")
            record = AssetRecord(
                asset_id=str(raw["id"]),
                kind=str(raw["kind"]),
                status=str(raw.get("status", "verified")),
                files=dict(raw.get("files", {})),
                sha256=dict(raw.get("sha256", {})),
                metadata=dict(raw.get("metadata", {})),
            )
            if record.asset_id in self._records:
                raise ValueError(f"duplicate asset id in {self.path}: {record.asset_id}")
            self._records[record.asset_id] = record

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AssetManifest":
        manifest_path = Path(path) if path else default_asset_manifest_path()
        with manifest_path.expanduser().resolve().open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"asset manifest must be a JSON object: {manifest_path}")
        return cls(manifest_path, payload)

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def get(self, asset_id: str) -> AssetRecord:
        try:
            return self._records[asset_id]
        except KeyError as exc:
            raise KeyError(f"asset is not registered in {self.path}: {asset_id}") from exc

    def logical_path(self, path: str | Path) -> Path:
        """Map a physical asset to its declared checkout-relative path."""
        resolved = (self.root / Path(path).expanduser()).resolve()
        if resolved.is_relative_to(self.root):
            return resolved.relative_to(self.root)
        for alias, external_root in self.external_roots:
            if resolved.is_relative_to(external_root):
                return alias / resolved.relative_to(external_root)
        raise ValueError(f"asset is outside declared roots: {resolved}")

    def resolve(
        self,
        asset_id: str,
        file_key: str = "default",
        *,
        allow_legacy: bool = False,
        require_exists: bool = True,
        verify_hash: bool = True,
    ) -> Path:
        record = self.get(asset_id)
        if record.status == "legacy" and not allow_legacy:
            raise ValueError(f"asset is legacy and cannot be used: {asset_id}")
        relative = record.files.get(file_key)
        if relative is None:
            raise KeyError(f"asset {asset_id} has no file key {file_key!r}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            logical = Path(relative)
            declared = (
                not logical.is_absolute()
                and ".." not in logical.parts
                and any(logical.is_relative_to(alias) and path.is_relative_to(external_root)
                        for alias, external_root in self.external_roots)
            )
            if not declared:
                raise ValueError(f"asset escapes manifest root: {asset_id} -> {relative}")
        if require_exists and not path.exists():
            raise FileNotFoundError(f"asset file is missing: {asset_id} -> {path}")
        expected = record.sha256.get(file_key)
        if verify_hash and expected and path.is_file() and sha256_file(path) != expected:
            raise ValueError(f"asset hash mismatch: {asset_id} ({file_key}) -> {path}")
        return path

    def validate(self, *, require_exists: bool = True, include_legacy: bool = False) -> list[str]:
        errors: list[str] = []
        for asset_id, record in self._records.items():
            if record.status == "legacy" and not include_legacy:
                continue
            for file_key in record.files:
                try:
                    self.resolve(
                        asset_id,
                        file_key,
                        allow_legacy=include_legacy,
                        require_exists=require_exists,
                    )
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    if isinstance(exc, FileNotFoundError) and record.metadata.get("allow_missing"):
                        continue
                    errors.append(str(exc))
        return errors


def default_asset_manifest_path() -> Path:
    from somaforge_core.robot_assets import somaforge_root

    return somaforge_root() / "configs" / "assets_manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
