"""Freeze bundles and protocol before generating a new synthetic weather path."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import sys

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
pv = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pv))
from pv_forecast.backtesting import backtest_bundles


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


generator_path = pv.parent / "数据" / "光伏数据" / "generate_huanghua_pv.py"
config_path = generator_path.parent / "simulation_config.json"
bundles = [pv / "运行结果" / name for name in
           ("20260929_000009_045346", "20261009_daylight_recalibration_01", "20261009_night_zero_lower_01")]
out = Path(__file__).parent / "frozen_2026_01"
assert not out.exists(), "new experiment directory required; inspect any existing/live run before retrying"
out.mkdir()
config = copy.deepcopy(read(config_path))
config.update(year=2026, seed=20261009, dataset_id="huanghua_pv_2026_frozen_eval_seed20261009_v1")
config["output"]["data_filename"] = "黄花站_2026年_15分钟_冻结评估模拟数据.csv"
hashes = {str(p.relative_to(pv)): digest(p) for p in sorted((pv / "pv_forecast").glob("*.py"))}
bundle_hashes = {bundle.name: {name: digest(bundle / name) for name in
                             ("model_metadata.json", "gru.keras", "scalers.joblib")}
                 for bundle in bundles}
protocol = {"status": "frozen_before_generation", "year": 2026, "seed": 20261009,
    "evaluation_scope": "all complete 3-day-history/24-hour-target windows, 15-minute stride",
    "weights_fitting_allowed": False, "calibration_fitting_allowed": False, "model_selection_allowed": False,
    "source_code_sha256": hashes, "bundle_sha256": bundle_hashes,
    "generator_sha256": digest(generator_path), "generator_config": config,
    "limitations": ["Fresh random path from the known simulator, not real or independently collected site data.",
                    "The zero-support rule was designed after inspecting 2025; 2025 remains retrospective.",
                    "Overlapping target windows are dependent; no future coverage guarantee."],
    "original_data_sha256": digest(generator_path.parent / "黄花站_2025年_15分钟光伏模拟数据.csv")}
# This write precedes importing/running the simulator or observing its values.
write(out / "protocol.json", protocol)
write(out / "simulation_config.json", config)
module_spec = importlib.util.spec_from_file_location("frozen_eval_simulator", generator_path)
simulator = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(simulator)
frame = simulator.simulate(config)
facts = simulator.validate_frame(frame, config)
payload = simulator.csv_bytes(frame, config)
assert payload == simulator.csv_bytes(simulator.simulate(config), config)
csv = out / config["output"]["data_filename"]
csv.write_bytes(payload)
monthly = simulator.monthly_summary(frame, config)
(out / "monthly_summary.csv").write_bytes(simulator.csv_bytes(monthly, config))
write(out / "validation_report.json", {"status": "passed", "is_simulated": True, "checks": facts,
    "repeat_generation_byte_identical": True, "source_sha256": digest(csv), "generator_sha256": digest(generator_path)})


def progress(message):
    print(message, file=sys.stderr, flush=True)
    with (out / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


report = backtest_bundles(bundles, csv, out / "backtest.json", batch_windows=1024, progress=progress)
assert report["evaluated_windows"] == len(frame)-288-96+1 and report["skipped_origins"] == 0
assert all(values["after_parent_input_window"] for values in report["models"].values())
assert all(digest(pv / name) == expected for name, expected in hashes.items())
assert digest(generator_path) == protocol["generator_sha256"]
assert digest(generator_path.parent / "黄花站_2025年_15分钟光伏模拟数据.csv") == protocol["original_data_sha256"]
protocol.update(status="completed", evaluated_windows=report["evaluated_windows"],
                backtest_ref="backtest.json", source_sha256=digest(csv))
write(out / "completion.json", protocol)
print(json.dumps({"status": "completed", "windows": report["evaluated_windows"],
    "models": {name: values["overall"] for name, values in report["models"].items()}}, indent=2))
