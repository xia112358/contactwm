"""Compatibility alias for :mod:`generator.next_interaction_joint`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("generator.next_interaction_joint", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("generator.next_interaction_joint")
