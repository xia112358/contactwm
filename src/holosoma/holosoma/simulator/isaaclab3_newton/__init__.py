__all__ = ["IsaacLab3Newton"]


def __getattr__(name):
    # Configuration and asset helpers can be used before the Kit runtime starts.
    if name == "IsaacLab3Newton":
        from .isaaclab3_newton import IsaacLab3Newton
        return IsaacLab3Newton
    raise AttributeError(name)
