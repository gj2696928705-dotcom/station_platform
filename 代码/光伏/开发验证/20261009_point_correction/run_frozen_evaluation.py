"""Freeze the final correction before generating one new random evaluation path."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import sys

os.environ.update(TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
out = Path(__file__).resolve().parent
pv = out.parents[1]
sys.path.insert(0, str(pv))
from pv_forecast.backtesting import backtest_bundles


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


candidate = pv/"运行结果"/"20261009_point_correction_01"
assert read(candidate/"point_correction_report.json")["status"] == "completed"
assert read(out/"development_completion.json")["status"] == "completed"
folder = out/"frozen_2026_seed20261010"
assert not folder.exists(), "new frozen evaluation directory required; inspect existing/live experiment"
folder.mkdir()
generator = pv.parent/"数据"/"光伏数据"/"generate_huanghua_pv.py"
config = copy.deepcopy(read(generator.parent/"simulation_config.json"))
config.update(year=2026, seed=20261010, dataset_id="huanghua_pv_point_correction_frozen_2026_seed20261010_v1")
config["output"]["data_filename"] = "黄花站_2026年_15分钟_点预测修正冻结模拟数据.csv"
bundles = [pv/"运行结果"/name for name in
           ("20260929_000009_045346", "20261009_night_zero_lower_01", "20261009_point_correction_01")]
protocol = {"status": "frozen_before_generation", "year": 2026, "seed": 20261010,
    "source_sha256": {p.relative_to(pv).as_posix(): sha(p) for p in sorted((pv/"pv_forecast").glob("*.py"))},
    "bundle_sha256": {b.name: {name: sha(b/name) for name in
        ("model_metadata.json", "gru.keras", "scalers.joblib", "point_correction.joblib") if (b/name).exists()} for b in bundles},
    "generator_sha256": sha(generator), "generator_config": config,
    "original_data_sha256": sha(generator.parent/"黄花站_2025年_15分钟光伏模拟数据.csv"),
    "no_fitting_or_selection_on_evaluation": True, "full_window_scoring": True,
    "planned_comparisons": ["daylight and all-day RMSE/MAE against the frozen raw GRU", "all 12 target months",
                            "saved interval coverage/width", "weather labels for diagnosis after prediction only"],
    "limitations": ["One new path of the known simulator; not independently collected real-site data.",
                    "Overlapping labels count by forecast origin and are dependent.",
                    "No production threshold established; report improvements and failures without claiming real accuracy."]}
write(folder/"protocol.json", protocol)
write(folder/"simulation_config.json", config)
module_spec = importlib.util.spec_from_file_location("point_eval_simulator", generator)
simulator = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(simulator)
frame = simulator.simulate(config)
facts = simulator.validate_frame(frame, config)
payload = simulator.csv_bytes(frame, config)
assert payload == simulator.csv_bytes(simulator.simulate(config), config)
source = folder/config["output"]["data_filename"]
source.write_bytes(payload)
write(folder/"validation_report.json", {"status": "passed", "is_simulated": True,
    "checks": facts, "repeat_generation_byte_identical": True, "source_sha256": sha(source)})
(folder/"monthly_summary.csv").write_bytes(simulator.csv_bytes(simulator.monthly_summary(frame, config), config))


def progress(message):
    print(message, file=sys.stderr, flush=True)
    with (folder/"progress.log").open("a", encoding="utf-8") as handle:
        handle.write(message+"\n")


report = backtest_bundles(bundles, source, folder/"backtest.json", batch_windows=1024, progress=progress)
assert report["evaluated_windows"] == 34657 and report["skipped_origins"] == 0
assert all(sha(pv/name) == expected for name, expected in protocol["source_sha256"].items())
assert all(sha(pv/"运行结果"/bundle/name) == expected for bundle, values in protocol["bundle_sha256"].items() for name, expected in values.items())
assert sha(generator) == protocol["generator_sha256"]
assert sha(generator.parent/"黄花站_2025年_15分钟光伏模拟数据.csv") == protocol["original_data_sha256"]
protocol.update(status="completed", evaluated_windows=report["evaluated_windows"],
                source_sha256=sha(source), backtest_ref="backtest.json")
write(folder/"completion.json", protocol)
print(json.dumps({"status": "completed", "windows": report["evaluated_windows"],
    "models": {name: model["overall"] for name, model in report["models"].items()}}, indent=2))
