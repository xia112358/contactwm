"""Compatibility alias for :mod:`generator.fullbody_dataset`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("generator.fullbody_dataset", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("generator.fullbody_dataset")
