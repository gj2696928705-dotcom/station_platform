"""Package pinned candidate code/weights without modifying any source model.

Creates a new destination; never selects a latest model, trains, or overwrites.
Numerical interpreter is configured separately from the platform interpreter.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

from capacity_alignment.forecast_adapter import sha
from capacity_alignment.common import digest


def package(kind, code_dir, bundle_dir, python, output):
    code_dir, bundle_dir, output = Path(code_dir).resolve(), Path(bundle_dir).resolve(), Path(output).resolve()
    module = "load_forecast" if kind == "load" else "pv_forecast"
    required = ["model_metadata.json", "interval_calibration.json"] + (["xgboost.json"] if kind == "load" else ["gru.keras", "scalers.joblib"])
    for name in required:
        if not (bundle_dir/name).is_file(): raise ValueError("Missing pinned model file: " + name)
    metadata = json.loads((bundle_dir/"model_metadata.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(code_dir/module, output/"source"/module, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output/"bundle").mkdir()
    for name in required:
        shutil.copy2(bundle_dir/name, output/"bundle"/name)
    for name in ["requirements.txt", "README.md"]:
        shutil.copy2(code_dir/name, output/name)
    # Record the actual independent runtime, including native libraries, as a lock.
    check = "import importlib.metadata as m,json,sys; import " + ("xgboost" if kind == "load" else "tensorflow") + "; print(json.dumps({'python':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    runtime = json.loads(subprocess.check_output([str(python), "-X", "utf8", "-c", check], timeout=60))
    (output/"runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    packages = sorted(runtime["packages"].items())
    (output/"requirements-lock.txt").write_text("\n".join(f"{p}=={v}" for p,v in packages)+"\n", encoding="utf-8")
    files = {path.relative_to(output).as_posix(): sha(path) for path in output.rglob("*") if path.is_file()}
    manifest = {"schema_version": "candidate-deployment-1", "simulation": True, "kind": kind,
                "station_id": "HH-Huanghua-CSS", "meter_boundary": "M-PCC/AC-BUS",
                "binding_evidence": "synthetic Huanghua data and registry; no real site binding",
                "python": str(Path(python).resolve()), "model_version": metadata["model_version"],
                "value_semantics": "interval_average_power" if kind == "load" else "hourly_mean_start_label",
                "publication_status": "candidate", "real_dispatch_eligible": False,
                "original_code_dir": str(code_dir), "original_bundle_dir": str(bundle_dir), "files": files}
    manifest["manifest_hash"] = digest(manifest)
    (output/"deployment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["load", "pv"], required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.kind, args.code_dir, args.bundle_dir, args.python, args.output)
    print(json.dumps({"manifest_hash": result["manifest_hash"], "file_count": len(result["files"])}))


if __name__ == "__main__":
    main()
