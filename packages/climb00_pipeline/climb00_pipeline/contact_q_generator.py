"""Compatibility alias for :mod:`generator.contact_q_generator`."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("generator.contact_q_generator", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("generator.contact_q_generator")
