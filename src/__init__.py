"""Payment-fraud scoring pipeline.

Environment thread caps are set here so they apply before NumPy and LightGBM import.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

__version__ = "1.0.0"
