"""Recover completed fits into a new run; restart only the interrupted trial."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
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
import joblib
import numpy as np
from pv_forecast.models import GRUForecaster
from pv_forecast.physics_tree import PhysicsTreeForecaster
from pv_forecast.physics_experiments import compare_physics_models


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original = pv/"运行结果"/"20261014_physics_comparison_01"
destination = pv/"运行结果"/"20261015_physics_comparison_recovery_01"
assert not destination.exists() and not (out/"development_completion.json").exists()
protocol = read(out/"development_protocol.json")
assert all(sha(pv/name) == value for name, value in protocol["implementation_sha256"].items())
source_hashes = {p.relative_to(original).as_posix(): sha(p) for p in original.rglob("*") if p.is_file()}
assert not (original/"experiments"/"month_09"/"physics_gru16_scheduled"/"training_summary.json").exists()
recovery = {"status": "registered_before_recovery", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_run": str(original), "recovery_run": str(destination),
    "reason": "original training process exited during month_09/gru16_scheduled; eight complete GRU fits exist",
    "original_artifact_sha256": source_hashes, "reused_gru_fits": [], "reused_tree_fits": [],
    "interrupted_partial_history_epochs": len(read(original/"experiments/month_09/physics_gru16_scheduled/training_history.json")),
    "interrupted_trial_policy": "retain partial evidence; restart only this trial with all windows and original full config",
    "sample_cap": None, "implementation_changed": False, "recovery_script_sha256": sha(Path(__file__))}
write(out/"recovery_protocol.json", recovery)
write(out/"active_comparison.json", {"path": str(destination), "original_path": str(original), "recovery_protocol": "recovery_protocol.json"})
original_gru_fit, original_gru_save, original_tree_fit = GRUForecaster.fit, GRUForecaster.save, PhysicsTreeForecaster.fit


def previous_folder(directory):
    relative = Path(directory).resolve().relative_to(destination.resolve())
    return original/relative


def fit_gru(self, x_hist, x_future, y, validation_data=None, *, target_scaler=None, output_dir=None, progress=None):
    folder = Path(output_dir)
    previous = previous_folder(folder)
    if (previous/"training_summary.json").exists():
        summary = read(previous/"training_summary.json")
        history = read(previous/"training_history.json")
        config = read(previous/"training_config.json")
        assert summary["status"] == "completed" and summary["training_windows_used"] == len(y)
        assert summary["history_feature_count"] == x_hist.shape[-1] == 13
        assert config["gru_epochs"] == self.config.epochs and config["gru_units"] == self.config.units
        assert summary["epochs_completed"] == len(history)
        assert read(previous/"data_split.json")["validation_windows"] == len(validation_data[2])
        loaded = GRUForecaster.load(str(previous/"gru.keras"), config=self.config)
        self.model, self.history, self.training_summary = loaded.model, history, summary
        self._recovery_source_checkpoint = previous/"gru.keras"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("training_history.json", "training_history.csv", "training_summary.json", "training.log", "gru.keras"):
            shutil.copy2(previous/name, folder/name)
        shutil.copytree(previous/"visualizations", folder/"visualizations", dirs_exist_ok=True)
        recovery["reused_gru_fits"].append({"path": str(previous), "sha256": sha(previous/"gru.keras"),
            "training_windows": len(y), "epochs": len(history), "best_epoch": summary["best_epoch"]})
        write(out/"recovery_progress.json", recovery)
        if progress:
            progress(f"复用完整训练：{len(history)}轮、最佳{summary['best_epoch']}轮、全部{len(y)}窗口；原权重不变。")
        return self
    if progress and previous.exists():
        progress(f"仅重新完整训练中断候选；原部分检查点保留在{previous}，全部{len(y)}窗口。")
    return original_gru_fit(self, x_hist, x_future, y, validation_data=validation_data,
        target_scaler=target_scaler, output_dir=output_dir, progress=progress)


def save_gru(self, path):
    if hasattr(self, "_recovery_source_checkpoint"):
        shutil.copy2(self._recovery_source_checkpoint, path)
        assert sha(Path(path)) == sha(self._recovery_source_checkpoint)
        return
    return original_gru_save(self, path)


def fit_tree(self, train, validation, *, capacity_kw, output_dir=None, progress=None):
    folder = Path(output_dir)
    previous = previous_folder(folder)
    if (previous/"physics_tree.joblib").exists():
        loaded = joblib.load(previous/"physics_tree.joblib")
        assert loaded.window_count == train.size and asdict(loaded.config) == asdict(self.config)
        self.__dict__.update(loaded.__dict__)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("training_history.json", "training_summary.json"):
            shutil.copy2(previous/name, folder/name)
        recovery["reused_tree_fits"].append({"path": str(previous), "training_windows": train.size,
            "sha256": sha(previous/"physics_tree.joblib")})
        write(out/"recovery_progress.json", recovery)
        if progress:
            progress(f"复用完整树训练：{self.config.iterations}轮，全部{train.size}窗口。")
        return self
    return original_tree_fit(self, train, validation, capacity_kw=capacity_kw, output_dir=output_dir, progress=progress)


GRUForecaster.fit, GRUForecaster.save, PhysicsTreeForecaster.fit = fit_gru, save_gru, fit_tree
source = pv.parent/"数据"/"光伏数据"/"黄花站_2025年_15分钟光伏模拟数据.csv"
control = pv/"运行结果"/"20261009_point_correction_01"
report = compare_physics_models(source, control, source.parent/"simulation_config.json", destination,
    batch_windows=1024, progress=lambda message: print(message, file=sys.stderr, flush=True))
assert report["status"] == "completed" and len(recovery["reused_gru_fits"]) == 8 and len(recovery["reused_tree_fits"]) == 3
assert report["final_training_windows"] == 24145 and report["retrospective_test_windows"] == 5161
assert all(sha(original/name) == value for name, value in source_hashes.items())
assert sha(source) == protocol["source_sha256"]
assert all(sha(control/name) == value for name, value in protocol["control_sha256"].items())
recovery.update(status="completed", original_artifacts_unchanged=True)
write(out/"recovery_completion.json", recovery)
protocol.update(status="completed", report_ref=str((destination/"comparison_report.json").relative_to(pv)),
    selected_candidate=report["selected_candidate"], best_enhanced_gru=report["best_enhanced_gru"],
    recovery_protocol_ref="recovery_protocol.json", recovery_completion_ref="recovery_completion.json")
write(out/"development_completion.json", protocol)
print(json.dumps({"status": "completed", "selected": report["selected_candidate"], "aggregate": report["aggregate"]}, indent=2))
