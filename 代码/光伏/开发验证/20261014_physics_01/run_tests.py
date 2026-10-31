"""Run unchanged unittest discovery with an approved workspace temp directory."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

phase = Path(__file__).resolve().parent
pv = phase.parents[1]
scratch = (phase/"test_tmp").resolve()
assert scratch.is_relative_to(pv.resolve())
scratch.mkdir(exist_ok=True)
tempfile.tempdir = str(scratch)
# Process-local redirection also covers Keras temporary files in child CLIs.
os.environ.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
sys.path.insert(0, str(pv))
pattern = sys.argv[1] if len(sys.argv) > 1 else "test*.py"
suite = unittest.defaultTestLoader.discover(str(pv/"tests"), pattern=pattern)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
