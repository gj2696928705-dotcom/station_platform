"""Independent recheck of PV and isolated current-platform regressions."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

out = Path(__file__).resolve().parent
pv = out.parents[1]
workspace = pv.parents[1]
scratch = out/"runtime_tmp"
scratch.mkdir(exist_ok=True)
os.environ.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1",
    TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
tempfile.tempdir = str(scratch)
sys.path.insert(0, str(workspace))
if sys.argv[1] == "pv":
    sys.path.insert(0, str(pv))
    suite = unittest.defaultTestLoader.discover(str(pv/"tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
site = workspace/"station_platform"/".venv"/"Lib"/"site-packages"
sys.path.insert(0, str(site))
import pytest
raise SystemExit(pytest.main([str(workspace/"station_platform"/"tests"), "-q",
    "--basetemp", str(out/"pytest_tmp"), "--junitxml", str(out/"platform_tests.xml")]))
