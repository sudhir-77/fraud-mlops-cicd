"""Streaming score path, CI gate, and deployment checks for the fraud model.

Thread caps are set before NumPy loads so a `python -m streaming.ci` process
stays inside the same budget as `python run.py`.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

__version__ = "1.0.0"
