"""Compatibility alias for :mod:`generator.conditioned_pose_predictor`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("generator.conditioned_pose_predictor", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("generator.conditioned_pose_predictor")
