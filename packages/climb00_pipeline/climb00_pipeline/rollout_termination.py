"""Compatibility alias for :mod:`generator.rollout_termination`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("generator.rollout_termination", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("generator.rollout_termination")
