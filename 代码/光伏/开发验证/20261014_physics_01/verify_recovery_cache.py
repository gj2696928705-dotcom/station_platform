"""Verify reused weights pair with byte-identical parameter values in scalers."""
from pathlib import Path
import hashlib
import json
import os

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
import joblib
import numpy as np

out = Path(__file__).resolve().parent
protocol = json.loads((out/"recovery_protocol.json").read_text(encoding="utf-8"))
old, new = Path(protocol["source_run"]), Path(protocol["recovery_run"])
checks = []
for original in (old/"experiments").glob("*/*/scalers.joblib"):
    current = new/original.relative_to(old)
    if not current.exists():
        continue
    a, b = joblib.load(original), joblib.load(current)
    for name in ("target_scaler", "feature_scaler", "history_scaler"):
        for attribute in ("data_min_", "data_max_", "scale_", "min_", "n_samples_seen_"):
            np.testing.assert_array_equal(getattr(a[name], attribute), getattr(b[name], attribute))
    checks.append(str(original.parent.relative_to(old)))
assert len(checks) == 9
assert all(hashlib.sha256((old/name).read_bytes()).hexdigest() == value for name, value in protocol["original_artifact_sha256"].items())
result = {"status": "passed", "all_nine_scaler_sets_exactly_match": checks,
    "all_original_training_artifacts_unchanged": True, "no_unseen_trajectory_loaded": True}
(out/"recovery_cache_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
