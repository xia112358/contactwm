"""Compatibility alias for :mod:`contact_solver.contact_validity`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("contact_solver.contact_validity", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("contact_solver.contact_validity")
