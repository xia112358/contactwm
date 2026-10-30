"""Compatibility alias for :mod:`generator.demonstration_teacher`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("generator.demonstration_teacher", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("generator.demonstration_teacher")
