"""Compatibility alias for :mod:`generator.dagger_supervision`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("generator.dagger_supervision", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("generator.dagger_supervision")
