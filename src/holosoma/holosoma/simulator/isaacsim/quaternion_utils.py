"""Conversions at configuration/USD boundaries; simulation states remain XYZW."""

from collections.abc import Sequence


def config_wxyz_to_xyzw(rotation: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert an explicitly WXYZ configuration value to Isaac Lab 3 order.

    Do not apply this to root/body states, motion tensors, or checkpoint observations:
    those already use XYZW. USD's Gf quaternion constructors still take WXYZ.
    """
    if len(rotation) != 4:
        raise ValueError(f"Expected four WXYZ components, received {len(rotation)}")
    w, x, y, z = rotation
    return (x, y, z, w)


def depth_rotation_to_xyzw(rotation: Sequence[float], convention: str) -> tuple[float, float, float, float]:
    """Preserve old PHP camera frames or interpret an explicitly WXYZ config."""
    if convention == "wxyz":
        return config_wxyz_to_xyzw(rotation)
    if convention != "legacy_xyzw":
        raise ValueError(f"Unknown depth rotation convention: {convention}")
    if len(rotation) != 4:
        raise ValueError("Expected four legacy XYZW components")
    return tuple(rotation)
