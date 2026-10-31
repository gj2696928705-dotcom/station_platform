"""One-shot continuation of this full run; no retraining or repeat generation."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
pv = out.parents[1]
assert not (out/"candidate_cli").exists() and not (out/"frozen_three_paths").exists()
scratch = (out/"runtime_tmp").resolve()
assert scratch.is_relative_to(pv.resolve())
scratch.mkdir(exist_ok=True)
env = os.environ.copy()
env.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
while not (out/"development_completion.json").exists():
    if "Traceback" in (out/"comparison_runner.log").read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("full comparison failed; inspect existing artifacts before continuing")
    time.sleep(10)
completion = json.loads((out/"development_completion.json").read_text(encoding="utf-8"))
assert completion["status"] == "completed"
for phase in ("verify_candidate_cli", "run_frozen_cohort", "aggregate_frozen", "audit_evidence"):
    print(f"Starting {phase}", flush=True)
    with (out/(phase+"_stdout.log")).open("w", encoding="utf-8") as stdout, (out/(phase+"_stderr.log")).open("w", encoding="utf-8") as stderr:
        result = subprocess.run([sys.executable, "-X", "utf8", str(out/(phase+".py"))], cwd=pv, env=env,
            stdout=stdout, stderr=stderr)
    assert result.returncode == 0, f"{phase} failed; inspect logs; do not regenerate viewed paths"
    print(f"Completed {phase}", flush=True)
print("All candidate CLI, frozen cohorts, aggregate and evidence checks completed; review PNGs and finalize docs.", flush=True)
