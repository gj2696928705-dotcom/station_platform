"""Reproducible local checks with a writable temporary directory."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

out = Path(__file__).resolve().parent
pv = out.parents[1]
scratch = out/"runtime_tmp"
scratch.mkdir(exist_ok=True)
os.environ.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1",
    TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
tempfile.tempdir = str(scratch)
sys.path.insert(0, str(pv))
if sys.argv[1] == "train":
    import runpy
    runner = runpy.run_path(str(pv/"运行.py"))
    raise SystemExit(runner["main"](["--output-dir", str(pv/"运行结果"/"20261026_day13_01")]))
pattern = "test_day13.py" if sys.argv[1] == "focused" else "test*.py"
suite = unittest.defaultTestLoader.discover(str(pv/"tests"), pattern=pattern)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
