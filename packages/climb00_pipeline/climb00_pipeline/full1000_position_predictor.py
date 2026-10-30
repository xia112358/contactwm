"""Compatibility alias for :mod:`generator.full1000_position_predictor`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("generator.full1000_position_predictor", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("generator.full1000_position_predictor")
