"""Full causal physics/tree/GRU comparisons with frozen legacy controls."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .baselines import BaselineForecaster
from .data import InputSpec, _read_source, _apply_column_map, _normalise_timestamp, build_windows, load_and_regularize, split_boundaries
from .diagnostics import ForecastDiagnostics, PointStats
from .experiments import default_folds, fold_bounds, candidate_configs
from .metrics import calibrate_prediction_interval
from .models import GRUForecaster
from .physics import load_physics_config, enhance_windows, clear_persistence
from .physics_tree import PhysicsTreeForecaster, PhysicsTreeConfig
from .pipeline import (TrainingConfig, _prediction_spec, _json_dump, _load_bundle, _fit_scalers,
                       _fit_history_scaler, _scale_windows, _gru_config, _daylight_mask, _candidate_baselines)
from .point_correction import SolarWeatherRidge
from .point_experiments import _batched_prediction
from .recalibration import _predict_windows, _scores


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_physics_models(source, control_bundle, physics_source, output_dir, *,
                           config=TrainingConfig(require_gru=True, interval_conditioning="daylight", night_lower_bound="zero"),
                           tree_config=PhysicsTreeConfig(), folds=None, batch_windows=1024, progress=None):
    """Use all rolling windows; test is reported only after selection/calibration.

    A legacy GRU/Ridge is reproduced from frozen rolling checkpoints. New
    historical/solar features are identical for trees and enhanced GRUs. No
    future meteorology or target weather labels enter the train/predict inputs.
    """
    output, control_dir = Path(output_dir).resolve(), Path(control_bundle).resolve()
    if output.exists() or output.is_relative_to(control_dir):
        raise ValueError("comparison requires a new directory outside the control bundle")
    if config.skip_gru or config.max_gru_samples or config.interval_minutes != 15 or config.history_days != 3 or config.horizon_steps != 96:
        raise ValueError("physics comparison requires full GRU training, 288 history and 96 horizon points at 15 minutes")
    if not isinstance(batch_windows, int) or isinstance(batch_windows, bool) or batch_windows < 1:
        raise ValueError("batch_windows must be a positive integer")
    control_meta, control_gru, control_scalers = _load_bundle(control_dir, allow_fallback=False)
    if control_meta["selected_model"] != "gru" or control_meta.get("physics_config"):
        raise ValueError("comparison control must be the unenhanced GRU or GRU/Ridge candidate")
    physics = load_physics_config(physics_source)
    spec = _prediction_spec(control_meta, InputSpec())
    if spec.weather_columns or (physics["station"]["latitude_deg"], physics["station"]["longitude_deg"]) != (spec.latitude, spec.longitude):
        raise ValueError("requires historical-only inputs and the control's configured solar coordinates")
    raw_parent = Path(control_meta.get("parent_bundle_ref", control_dir)).resolve()
    rolling_path = raw_parent/"rolling_validation.json"
    parent_rolling = json.loads(rolling_path.read_text(encoding="utf-8"))
    if parent_rolling["status"] != "completed" or parent_rolling["scope"] != "full_windows":
        raise ValueError("complete full-window legacy rolling training is required")
    control_candidate = control_meta["selected_gru_candidate"]
    alpha = (control_meta.get("point_correction") or {}).get("alpha")
    hashes = {str(p): _sha(p) for p in [control_dir/"model_metadata.json", control_dir/"gru.keras", control_dir/"scalers.joblib", rolling_path]}
    if alpha is not None:
        hashes[str(control_dir/"point_correction.joblib")] = _sha(control_dir/"point_correction.joblib")
    for fold in parent_rolling["folds"]:
        for name in parent_rolling["candidates"]:
            folder = raw_parent/"experiments"/fold["name"]/name
            split = json.loads((folder/"data_split.json").read_text(encoding="utf-8"))
            if split["training_windows_used"] != split["training_windows_available"]:
                raise ValueError("legacy control includes capped GRU training")
            for filename in ("gru.keras", "scalers.joblib", "training_summary.json"):
                hashes[str(folder/filename)] = _sha(folder/filename)
    raw = _apply_column_map(_read_source(source), spec)
    timestamps = _normalise_timestamp(raw[spec.timestamp_col], spec.timezone)
    if "is_simulated" not in raw or not raw.is_simulated.astype(str).str.lower().eq("true").all():
        raise ValueError("this registered comparison is restricted to explicitly simulated data")
    canonical = replace(spec, column_map={})
    train_end, development_end = split_boundaries(len(raw), config.train_ratio, config.val_ratio)
    regular = load_and_regularize(raw, canonical, available_before=timestamps[development_end])
    if len(regular.frame) != development_end or regular.quality.interpolated_target_rows or regular.quality.interpolated_weather_rows:
        raise ValueError("comparison requires complete observed development windows without imputation")
    fold_list = default_folds(timestamps) if folds is None else list(folds)
    bounds = fold_bounds(timestamps, fold_list, development_end)
    if len(bounds) != 3:
        raise ValueError("comparison requires three chronological rolling folds")
    candidates = candidate_configs(config)
    candidate_names = ["current_gru_ridge", "clear_persistence", "physics_tree", *["physics_"+name for name in candidates]]
    meters = {name: PointStats() for name in candidate_names}
    maes = {name: {} for name in candidate_names}
    weather_labels = raw.weather_type.astype(str).to_numpy() if "weather_type" in raw else None
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "all available 15-minute windows",
        "input_policy": "unchanged historical power/SWGDN/temperature/cloud, future solar/calendar only",
        "physics_config": physics, "gru_candidates": {k: asdict(v) for k, v in candidates.items()},
        "tree_config": asdict(tree_config), "control_bundle": str(control_dir), "control_hashes": hashes,
        "test_used_for_selection": False, "oracle_used_for_selection": False,
        "selection_metric": "pooled rolling daylight RMSE, then MAE; report weather and worst-fold counterexamples",
        "reserved_test_start": timestamps[development_end].isoformat(), "folds": []}
    _json_dump(output/"comparison_report.json", report)

    def emit(message):
        with (output/"training.log").open("a", encoding="utf-8") as handle:
            handle.write(message+"\n")
        if progress:
            progress(message)

    def build(lo, hi):
        return build_windows(regular, lo, hi, target_col=spec.target_col, history_weather_columns=spec.history_weather_columns,
                             latitude=spec.latitude, longitude=spec.longitude)

    def metadata(family, enhanced):
        value = copy.deepcopy(control_meta)
        value.update(selected_model=family, point_correction=None, physics_config=physics,
            raw_feature_names=list(control_meta["feature_names"]), raw_history_feature_names=list(control_meta["history_feature_names"]),
            feature_names=list(enhanced.feature_names), history_feature_names=list(enhanced.history_feature_names),
            model_version="0.3.0-physics-"+family+"-candidate", lifecycle_operation="physics_model_training",
            rolling_validation_ref=str(output/"comparison_report.json"), parent_bundle_ref=str(control_dir),
            publication_status="candidate", training_summary_ref="training_summary.json",
            selection_source="pooled_rolling_daylight_rmse", selected_gru_candidate=None)
        return value

    def annotate_scores(truth, prediction, windows):
        day = _daylight_mask(windows.forecast_starts, 96, 15, spec.latitude, spec.longitude)
        d = ForecastDiagnostics()
        positions = timestamps.get_indexer(windows.forecast_starts)
        ids = positions[:, None]+np.arange(96)[None]
        labels = weather_labels[ids] if weather_labels is not None else None
        previous = weather_labels[positions-1] if weather_labels is not None else None
        d.add(truth, prediction, day, labels, previous)
        overall = PointStats()
        overall.add(truth, prediction)
        return {"all_day": overall.result(), "diagnostics": d.result()}

    for fold, lo, hi in bounds:
        if lo*15/1440 < config.min_gru_days:
            raise ValueError("GRU fold has insufficient training days")
        train_raw, val_raw = build(0, lo), build(lo, hi)
        if not train_raw.size or not val_raw.size:
            raise ValueError("all training and validation partitions require complete windows")
        folder = raw_parent/"experiments"/fold.name/control_candidate
        legacy_split = json.loads((folder/"data_split.json").read_text(encoding="utf-8"))
        if legacy_split["training_windows_available"] != train_raw.size or legacy_split["validation_windows"] != val_raw.size:
            raise ValueError("new folds must reproduce the legacy control's complete windows")
        emit(f"[{fold.name}] 全部训练{train_raw.size}、验证{val_raw.size}窗口；复用原GRU权重建立对照。")
        legacy_model, legacy_scalers = GRUForecaster.load(str(folder/"gru.keras")), joblib.load(folder/"scalers.joblib")
        legacy_meta = copy.deepcopy(control_meta)
        legacy_meta["point_correction"] = None
        if alpha is not None:
            training_pred = _batched_prediction(train_raw, legacy_meta, legacy_model, legacy_scalers, batch_windows)
            legacy_model.point_correction = SolarWeatherRidge(alpha).fit(train_raw.gru_history, train_raw.future_features,
                training_pred, train_raw.target, train_raw.history_feature_names, train_raw.feature_names, batch_windows=batch_windows)
            del training_pred
        control_pred = _batched_prediction(val_raw, legacy_meta, legacy_model, legacy_scalers, batch_windows)
        del legacy_model, legacy_scalers
        train, val = enhance_windows(train_raw, physics), enhance_windows(val_raw, physics)
        target_scaler, feature_scaler = _fit_scalers(train)
        scalers = {"target_scaler": target_scaler, "feature_scaler": feature_scaler, "history_scaler": _fit_history_scaler(train)}
        results = {}
        day = _daylight_mask(val.forecast_starts, 96, 15, spec.latitude, spec.longitude)

        def record(name, prediction):
            values = annotate_scores(val.target, prediction, val)
            results[name] = values
            meters[name].add(val.target, prediction, day)
            if weather_labels is not None:
                pos = timestamps.get_indexer(val.forecast_starts)[:, None]+np.arange(96)[None]
                for state in np.unique(weather_labels[pos]):
                    maes[name].setdefault(state, PointStats()).add(val.target, prediction, day & (weather_labels[pos] == state))
            m = values["diagnostics"]["daylight"]
            emit(f"[{fold.name}/{name}] 白天RMSE={m['rmse']:.3f} kW MAE={m['mae']:.3f} R²={m['r2']:.4f}")
        record("current_gru_ridge", control_pred)
        record("clear_persistence", clear_persistence(val, physics))
        tree_dir = output/"experiments"/fold.name/"physics_tree"
        tree = PhysicsTreeForecaster(tree_config).fit(train, val, capacity_kw=physics["station"]["capacity_kw"],
            output_dir=tree_dir, progress=lambda msg: emit(f"[{fold.name}/physics_tree] {msg}"))
        joblib.dump(tree, tree_dir/"physics_tree.joblib")
        record("physics_tree", tree.predict(val.gru_history, val.future_features))
        del tree
        scaled_train, scaled_val = _scale_windows(train, **scalers), _scale_windows(val, **scalers)
        for name, candidate in candidates.items():
            directory = output/"experiments"/fold.name/("physics_"+name)
            directory.mkdir(parents=True)
            _json_dump(directory/"training_config.json", asdict(candidate))
            _json_dump(directory/"data_split.json", {"training_windows_available": train.size, "training_windows_used": train.size,
                "validation_windows": val.size, "train_end_exclusive": timestamps[lo].isoformat(),
                "validation_end_exclusive": timestamps[hi].isoformat(), "scalers_fit_on": "this fold training windows only"})
            _json_dump(directory/"features.json", {"history": train.history_feature_names, "future": train.feature_names,
                "raw_history": train_raw.history_feature_names, "raw_future": train_raw.feature_names, "physics": physics})
            joblib.dump(scalers, directory/"scalers.joblib")
            model = GRUForecaster(config=_gru_config(candidate))
            def trace(msg):
                with (directory/"training.log").open("a", encoding="utf-8") as h:
                    h.write(msg+"\n")
                emit(f"[{fold.name}/{name}] {msg}")
            model.fit(scaled_train.gru_history, scaled_train.future_features, scaled_train.target,
                validation_data=(scaled_val.gru_history, scaled_val.future_features, scaled_val.target),
                target_scaler=target_scaler, output_dir=directory, progress=trace)
            model.save(str(directory/"gru.keras"))
            record("physics_"+name, _batched_prediction(val, metadata("gru", val), model, scalers, batch_windows))
            _json_dump(directory/"metrics.json", results["physics_"+name])
            del model
        report["folds"].append({"name": fold.name, "training_windows": train.size, "validation_windows": val.size,
            "train_end_exclusive": timestamps[lo].isoformat(), "validation_end_exclusive": timestamps[hi].isoformat(), "results": results})
        _json_dump(output/"comparison_report.json", report)
        del train_raw, val_raw, train, val, scaled_train, scaled_val, scalers, control_pred
    aggregate = {name: {"daylight": meter.result(), "weather": {state: value.result() for state, value in maes[name].items()},
        "worst_fold_daylight_rmse": max(f["results"][name]["diagnostics"]["daylight"]["rmse"] for f in report["folds"])}
        for name, meter in meters.items()}
    best_gru = min(candidates, key=lambda name: (aggregate["physics_"+name]["daylight"]["rmse"], aggregate["physics_"+name]["daylight"]["mae"]))
    selected = min(aggregate, key=lambda name: (aggregate[name]["daylight"]["rmse"], aggregate[name]["daylight"]["mae"]))
    report.update(status="selection_complete", aggregate=aggregate, selected_candidate=selected, best_enhanced_gru=best_gru)
    _json_dump(output/"comparison_report.json", report)
    emit(f"滚动白天指标选中{selected}，最佳增强GRU={best_gru}；开始全部最终训练窗口拟合。")
    middle = (train_end+development_end)//2
    final_train_raw, val_select_raw, calibration_raw = build(0, train_end), build(train_end, middle), build(middle, development_end)
    final_train, final_val = enhance_windows(final_train_raw, physics), enhance_windows(val_select_raw, physics)
    final_scalers = dict(zip(("target_scaler", "feature_scaler"), _fit_scalers(final_train)))
    final_scalers["history_scaler"] = _fit_history_scaler(final_train)
    tree_dir, gru_dir, clear_dir = [output/"bundles"/name for name in ("physics_tree", "enhanced_gru", "clear_persistence")]
    for folder in (tree_dir, gru_dir, clear_dir):
        folder.mkdir(parents=True)
    tree = PhysicsTreeForecaster(tree_config).fit(final_train, final_val, capacity_kw=physics["station"]["capacity_kw"],
        output_dir=tree_dir, progress=emit)
    joblib.dump(tree, tree_dir/"physics_tree.joblib")
    joblib.dump(final_scalers, gru_dir/"scalers.joblib")
    chosen_config = candidates[best_gru]
    _json_dump(gru_dir/"training_config.json", asdict(chosen_config))
    scaled_train, scaled_val = _scale_windows(final_train, **final_scalers), _scale_windows(final_val, **final_scalers)
    final_gru = GRUForecaster(config=_gru_config(chosen_config))
    final_gru.fit(scaled_train.gru_history, scaled_train.future_features, scaled_train.target,
        validation_data=(scaled_val.gru_history, scaled_val.future_features, scaled_val.target),
        target_scaler=final_scalers["target_scaler"], output_dir=gru_dir, progress=emit)
    final_gru.save(str(gru_dir/"gru.keras"))
    del scaled_train, scaled_val, final_train, final_val
    splits = {}
    for name, lo, hi, windows in (("train", 0, train_end, final_train_raw), ("validation", train_end, development_end, build(train_end, development_end)),
            ("selection_validation", train_end, middle, val_select_raw), ("calibration", middle, development_end, calibration_raw)):
        splits[name] = {"start": timestamps[lo].isoformat(), "end": timestamps[hi-1].isoformat(), "rows": hi-lo, "windows": windows.size}
    splits["test"] = copy.deepcopy(control_meta["data_splits"]["test"])
    _, baseline_candidates = _candidate_baselines(final_train_raw, val_select_raw, 96)
    baseline = min(baseline_candidates, key=lambda item: item[0])[1]
    calibration_day = _daylight_mask(calibration_raw.forecast_starts, 96, 15, spec.latitude, spec.longitude)
    baseline_pred = baseline.predict(calibration_raw.history_power, 96)
    fallback = {"strategy": baseline.strategy, "calibration": calibrate_prediction_interval(calibration_raw.target,
        baseline_pred, config.nominal_coverage, daylight_mask=calibration_day, night_lower_bound="zero"),
        "audit_ref": str(output/"comparison_report.json")}
    bundles = {}
    from .physics import ClearPersistenceForecaster
    for family, folder, model, scalers in (("physics_tree", tree_dir, tree, None), ("gru", gru_dir, final_gru, final_scalers),
        ("physics_persistence", clear_dir, ClearPersistenceForecaster(physics, enhance_windows(calibration_raw, physics).history_feature_names,
                                                       enhance_windows(calibration_raw, physics).feature_names), None)):
        enhanced_cal = enhance_windows(calibration_raw, physics)
        meta = metadata(family, enhanced_cal)
        if family == "physics_tree":
            meta["physics_tree_sha256"] = _sha(folder/"physics_tree.joblib")
        if family == "gru":
            meta["selected_gru_candidate"] = best_gru
        meta.update(data_splits=splits, baseline_fallback=fallback, config=asdict(chosen_config),
            calibration_independent_of_final_early_stopping=True,
            test_usage="retrospective 2025 report; prospective evaluation uses new frozen paths")
        pred = _batched_prediction(calibration_raw, meta, model, scalers, batch_windows)
        meta["calibration"] = calibrate_prediction_interval(calibration_raw.target, pred, config.nominal_coverage,
            daylight_mask=calibration_day, night_lower_bound="zero")
        _json_dump(folder/"model_metadata.json", meta)
        _json_dump(folder/"interval_calibration.json", meta["calibration"])
        _json_dump(folder/"data_split.json", splits)
        bundles[folder.name] = {"path": str(folder), "metadata": meta, "model": model, "scalers": scalers}
    report.update(status="calibration_frozen", data_splits=splits, calibration_used_for_selection=False,
        final_training_windows=final_train_raw.size, final_selection_windows=val_select_raw.size,
        final_calibration_windows=calibration_raw.size,
        bundle_hashes={name: {p.name: _sha(p) for p in Path(value["path"]).iterdir()
            if p.name in ("model_metadata.json", "gru.keras", "scalers.joblib", "physics_tree.joblib", "interval_calibration.json")}
            for name, value in bundles.items()})
    _json_dump(output/"selection_and_calibration_frozen.json", report)
    del regular, final_train_raw, val_select_raw, calibration_raw, control_gru, control_scalers
    all_regular = load_and_regularize(raw, canonical)
    test = build_windows(all_regular, development_end, len(all_regular.frame), target_col=spec.target_col,
        history_weather_columns=spec.history_weather_columns, latitude=spec.latitude, longitude=spec.longitude)
    if test.size != splits["test"]["windows"]:
        raise ValueError("complete original retrospective test windows could not be reproduced")
    test_day = _daylight_mask(test.forecast_starts, 96, 15, spec.latitude, spec.longitude)
    test_reports = {}
    for name, value in bundles.items():
        pred = _batched_prediction(test, value["metadata"], value["model"], value["scalers"], batch_windows)
        test_reports[name] = {"scores": _scores(test, pred, value["metadata"]["calibration"], test_day),
                              "error_diagnostics": annotate_scores(test.target, pred, test)["diagnostics"]}
        _json_dump(Path(value["path"])/"metrics.json", test_reports[name])
    control_meta, control_gru, control_scalers = _load_bundle(control_dir, allow_fallback=False)
    control_pred = _batched_prediction(test, control_meta, control_gru, control_scalers, batch_windows)
    test_reports["current_gru_ridge"] = {"scores": _scores(test, control_pred, control_meta["calibration"], test_day),
        "error_diagnostics": annotate_scores(test.target, control_pred, test)["diagnostics"]}
    report.update(status="completed", retrospective_test_windows=test.size, retrospective_test=test_reports,
        bundle_dirs={name: value["path"] for name, value in bundles.items()},
        source_sha256=_sha(source) if isinstance(source, (str, Path)) else hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest(),
        deployment_recommendation="unchanged control pending three frozen unseen simulation paths; oracle excluded")
    _json_dump(output/"comparison_report.json", report)
    if any(_sha(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("legacy weights or metadata changed during comparison")
    emit(f"完整物理对照结束；测试{test.size}窗口，候选待三条冻结新轨迹验收。")
    return report
