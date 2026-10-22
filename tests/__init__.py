"""Test package. Caps BLAS/OpenMP threads before numpy loads: the suite is many small fits, and one process per
core-count of threads makes parallel test runs on a shared machine thrash. Override with OMP_NUM_THREADS etc."""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
