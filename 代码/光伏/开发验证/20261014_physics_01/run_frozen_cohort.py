"""Freeze candidate choice first, then evaluate three new full simulated years."""
from pathlib import Path
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

out = Path(__file__).resolve().parent
pv = out.parents[1]
sys.path.insert(0, str(pv))
scratch = (out/"runtime_tmp").resolve()
assert scratch.is_relative_to(pv.resolve())
scratch.mkdir(exist_ok=True)
tempfile.tempdir = str(scratch)
os.environ.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
from pv_forecast.backtesting import backtest_bundles
from pv_forecast.physics import load_physics_config


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


comparison_dir = Path(read(out/"active_comparison.json")["path"]) if (out/"active_comparison.json").exists() else pv/"运行结果"/"20261014_physics_comparison_01"
comparison = read(comparison_dir/"comparison_report.json")
assert comparison["status"] == "completed"
assert read(out/"development_completion.json")["status"] == "completed"
bundles = [pv/"运行结果"/"20261009_point_correction_01",
           *[Path(comparison["bundle_dirs"][name]) for name in ("physics_tree", "enhanced_gru", "clear_persistence")]]
folder = out/"frozen_three_paths"
assert not folder.exists(), "inspect existing/live cohort; never regenerate viewed paths"
folder.mkdir()
generator = pv.parent/"数据"/"光伏数据"/"generate_huanghua_pv.py"
base_config = read(generator.parent/"simulation_config.json")
seeds = read(out/"development_protocol.json")["unseen_evaluation_seeds"]
proposed = comparison["selected_candidate"]
proposal_bundle = ("20261009_point_correction_01" if proposed == "current_gru_ridge" else
                   "enhanced_gru" if proposed.startswith("physics_gru") else proposed)
protocol = {"status": "frozen_before_generation", "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "year": 2026, "seeds": seeds,
    "rolling_selected_candidate": proposed, "proposed_primary_bundle": proposal_bundle,
    "no_test_hyperparameter_tuning": True, "weights_fit_allowed": False, "calibration_fit_allowed": False,
    "input_policy": "historical power/weather and deterministic future solar/calendar; oracle separate",
    "source_code_sha256": {p.relative_to(pv).as_posix(): sha(p) for p in sorted((pv/"pv_forecast").glob("*.py"))},
    "experiment_script_sha256": {name: sha(out/name) for name in ("run_frozen_cohort.py", "aggregate_frozen.py")},
    "bundle_sha256": {b.name: {name: sha(b/name) for name in
        ("model_metadata.json", "gru.keras", "scalers.joblib", "point_correction.joblib", "physics_tree.joblib", "interval_calibration.json")
        if (b/name).exists()} for b in bundles},
    "generator_sha256": sha(generator), "base_generator_config": base_config,
    "original_data_sha256": sha(generator.parent/"黄花站_2025年_15分钟光伏模拟数据.csv"),
    "adoption_rule": {"only_rolling_selected_bundle_can_be_recommended": True,
        "each_path": "daylight RMSE and MAE lower than current control",
        "pooled_paths": "clear and partly-cloudy daylight RMSE must not worsen; worst target-month daylight RMSE must not worsen",
        "short_leads": "pooled 0-1h and 1-3h daylight RMSE and MAE must not worsen",
        "short_lead_guard_basis": "May development counterexample: scheduled enhanced GRU 0-1h RMSE 432.62 vs control 203.12 kW; added before any new path generation",
        "if_rule_fails": "retain current control and report limitations; do not select another model by test scores"},
    "limitations": ["Three new random paths of the same known simulator, not real-site acceptance.",
                    "Overlapping labels are dependent; scores count separately per forecast origin.",
                    "Oracle observed future weather is diagnostic only and cannot enter adoption or forecasting."]}
write(folder/"protocol.json", protocol)
assert read(out/"adoption_guard_amendment.json")["new_data_directory_existed"] is False
module_spec = importlib.util.spec_from_file_location("physics_cohort_simulator", generator)
simulator = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(simulator)
for seed in seeds:
    seed_dir = folder/str(seed)
    seed_dir.mkdir()
    config = copy.deepcopy(base_config)
    config.update(year=2026, seed=seed, dataset_id=f"huanghua_pv_physics_frozen_2026_seed{seed}_v1")
    config["output"]["data_filename"] = f"黄花站_2026年_物理对照_seed{seed}.csv"
    write(seed_dir/"simulation_config.json", config)
    frame = simulator.simulate(config)
    checks = simulator.validate_frame(frame, config)
    payload = simulator.csv_bytes(frame, config)
    assert payload == simulator.csv_bytes(simulator.simulate(config), config)
    source = seed_dir/config["output"]["data_filename"]
    source.write_bytes(payload)
    write(seed_dir/"validation_report.json", {"status": "passed", "checks": checks,
        "repeat_generation_byte_identical": True, "source_sha256": sha(source), "is_simulated": True})
    def progress(message):
        print(f"[seed={seed}] {message}", file=sys.stderr, flush=True)
        with (seed_dir/"progress.log").open("a", encoding="utf-8") as handle:
            handle.write(message+"\n")
    report = backtest_bundles(bundles, source, seed_dir/"backtest.json", batch_windows=1024,
        oracle_diagnostic_config=load_physics_config(base_config), progress=progress)
    assert report["evaluated_windows"] == 34657 and report["skipped_origins"] == 0
    assert report["ideal_future_weather_diagnostic"]["model_selection_eligible"] is False
    assert all(sha(pv/path) == expected for path, expected in protocol["source_code_sha256"].items())
    assert all(sha(out/name) == expected for name, expected in protocol["experiment_script_sha256"].items())
    assert all(sha(pv/"运行结果"/"20261009_point_correction_01"/name) == value for name, value in protocol["bundle_sha256"]["20261009_point_correction_01"].items())
    for bundle in bundles:
        assert all(sha(bundle/name) == value for name, value in protocol["bundle_sha256"][bundle.name].items())
    write(seed_dir/"completion.json", {"status": "completed", "seed": seed, "evaluated_windows": 34657,
        "source_sha256": sha(source), "backtest_ref": "backtest.json"})
    print(json.dumps({"seed": seed, "models": {name: values["overall"]["daylight"] for name, values in report["models"].items()},
        "oracle": report["ideal_future_weather_diagnostic"]["error_diagnostics"]["daylight"]}), flush=True)
assert sha(generator) == protocol["generator_sha256"]
assert sha(generator.parent/"黄花站_2025年_15分钟光伏模拟数据.csv") == protocol["original_data_sha256"]
write(folder/"completion.json", {"status": "completed", "seeds": seeds, "evaluated_windows": 3*34657,
    "scored_labels_per_model": 3*34657*96, "protocol_ref": "protocol.json"})
