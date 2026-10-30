"""Shared official Isaac Lab argument parsing before project/tyro options."""
from __future__ import annotations

import argparse
import sys


def _add_launcher_arguments(parser: argparse.ArgumentParser) -> None:
    """Keep the project's explicit headless CLI across launcher API revisions."""
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    for flag in ("--headless", "--enable_cameras"):
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, action="store_true", default=False)


def _mark_launcher_intent(args: argparse.Namespace, argv: list[str]) -> None:
    args.headless_explicit = "--headless" in argv
    args.visualizer_explicit = any(
        token in ("--visualizer", "--viz") or token.startswith(("--visualizer=", "--viz="))
        for token in argv
    )


def parse_isaaclab_launcher_args(description: str) -> argparse.Namespace:
    """Parse IsaacLab launcher args before Tyro sees project config args."""

    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument("--_holosoma-launcher-parser", action="store_true", help=argparse.SUPPRESS)
    _add_launcher_arguments(parser)
    launcher_args, remaining_args = parser.parse_known_args()
    _mark_launcher_intent(launcher_args, sys.argv[1:])
    sys.argv = [sys.argv[0]] + fold_preset_tokens(remaining_args)
    return launcher_args


def fold_preset_tokens(tokens: list[str]) -> list[str]:
    """Fold IsaacLab preset selector tokens without importing isaaclab_tasks."""
    names: list[str] = []
    kept: list[str] = []
    for token in tokens:
        if "=" not in token:
            kept.append(token)
            continue
        key, val = token.split("=", 1)
        if key in {"physics", "renderer"}:
            stripped = val.strip()
            if stripped:
                names.append(stripped)
        elif key == "presets":
            names.extend(name.strip() for name in val.split(",") if name.strip())
        else:
            kept.append(token)

    if not names:
        return list(kept)

    seen: set[str] = set()
    deduped = [name for name in names if not (name in seen or seen.add(name))]
    return [f"presets={','.join(deduped)}", *kept]
