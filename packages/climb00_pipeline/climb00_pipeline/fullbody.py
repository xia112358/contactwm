"""Compatibility alias for :mod:`somaforge_core.motion_trajectory`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("somaforge_core.motion_trajectory", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("somaforge_core.motion_trajectory")
