"""Tiny launcher: install OTel shim then exec Monty's run.py with the
remaining argv. Kept as a separate file so run_benchmark.sh doesn't
need to embed Python in a heredoc.
"""
import runpy
import sys

import otel_shim  # noqa: F401  -- side-effect: installs OTel + patches Monty

# Drop our own script name so Hydra sees the same argv as `python run.py …`.
sys.argv[0] = "run.py"
runpy.run_path("/workspace/monty/run.py", run_name="__main__")
