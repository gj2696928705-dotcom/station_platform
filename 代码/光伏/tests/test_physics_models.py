import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from pv_forecast.backtesting import backtest_bundles
from pv_forecast.cli import build_parser
from pv_forecast.data import InputSpec, build_windows, load_and_regularize
from pv_forecast.diagnostics import ForecastDiagnostics
from pv_forecast.metrics import calibrate_prediction_interval
from pv_forecast.models import GRUConfig, GRUForecaster
from pv_forecast.physics import load_physics_config, clear_profile, solar_vectors, enhance_windows
from pv_forecast.physics_tree import PhysicsTreeConfig, PhysicsTreeForecaster
from pv_forecast.pipeline import (TrainingConfig, train_model, predict_record, _fit_scalers, _fit_history_scaler,
                                  _scale_windows, _daylight_mask)
from pv_forecast.recalibration import _predict_windows, recalibrate_bundle

CONFIG = Path(__file__).resolve().parents[2]/"数据"/"光伏数据"/"simulation_config.json"


def fixture():
    cfg = load_physics_config(CONFIG)
    stamps = pd.date_range("2025-06-01", periods=16*96, freq="15min", tz="Asia/Shanghai")
    p, ghi = clear_profile(stamps, cfg)
    ratio = .5+.15*np.sin(np.arange(len(stamps))/71)
    frame = pd.DataFrame({"timestamp": stamps, "pv_power_kw": p*ratio, "SWGDN": ghi*ratio,
        "DHI": ghi*ratio*.12, "DNI": np.divide(ghi*ratio*.88, np.maximum(solar_vectors(stamps, 30.9, 111.35)[2], 1e-6)),
        "temperature_c": 20., "cloud_cover": 1-ratio, "weather_type": np.where(ratio > .5, "clear", "rain"), "is_simulated": "true"})
    spec = InputSpec(latitude=30.9, longitude=111.35, history_weather_columns=("SWGDN", "temperature_c", "cloud_cover"))
    regular = load_and_regularize(frame, spec)
    def windows(lo, hi):
        return build_windows(regular, lo, hi, latitude=spec.latitude, longitude=spec.longitude,
                             history_weather_columns=spec.history_weather_columns)
    return cfg, frame, spec, regular, windows


def tree_bundle(root):
    cfg, frame, spec, regular, windows = fixture()
    result = train_model(frame, root, spec, TrainingConfig(skip_gru=True))
    meta = result["metadata"]
    train_rows = meta["data_splits"]["train"]["rows"]
    val_end = train_rows+meta["data_splits"]["validation"]["rows"]
    train, val = enhance_windows(windows(0, train_rows), cfg), enhance_windows(windows(train_rows, val_end), cfg)
    model = PhysicsTreeForecaster(PhysicsTreeConfig(iterations=3, log_every=2)).fit(train, val, capacity_kw=5000)
    path = root/"physics_tree.joblib"
    joblib.dump(model, path)
    meta.update(selected_model="physics_tree", physics_config=cfg, point_correction=None,
        raw_feature_names=meta["feature_names"], raw_history_feature_names=meta["history_feature_names"],
        feature_names=val.feature_names, history_feature_names=val.history_feature_names,
        physics_tree_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    day = _daylight_mask(val.forecast_starts, 96, 15, 30.9, 111.35)
    pred = model.predict(val.gru_history, val.future_features)
    meta["calibration"] = calibrate_prediction_interval(val.target, pred, daylight_mask=day, night_lower_bound="zero")
    (root/"model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return cfg, frame, spec, regular, windows, meta, model


class PhysicsTests(unittest.TestCase):
    def test_geometry_timezone_and_low_sun_ratio_boundary(self):
        cfg, frame, spec, regular, windows = fixture()
        a = clear_profile(regular.frame.index, cfg)
        b = clear_profile(regular.frame.index.tz_convert("UTC"), cfg)
        np.testing.assert_allclose(a[0], b[0], atol=0., rtol=0.)
        w = windows(0, 8*96)
        enhanced = enhance_windows(w, cfg)
        self.assertEqual(enhanced.size, w.size)
        np.testing.assert_array_equal(enhanced.target, w.target)
        self.assertTrue(np.isfinite(enhanced.gru_history).all())
        self.assertTrue(np.isfinite(enhanced.future_features).all())
        self.assertIs(enhance_windows(enhanced, cfg), enhanced)
        night = enhanced.gru_history[..., list(enhanced.history_feature_names).index("clear_pv_kw")] == 0
        ratio = enhanced.gru_history[..., list(enhanced.history_feature_names).index("power_clear_ratio")]
        np.testing.assert_array_equal(ratio[night], 0)

    def test_future_target_values_and_labels_cannot_enter_features(self):
        cfg, frame, spec, regular, windows = fixture()
        first = windows(0, 4*96)
        changed = copy.deepcopy(first)
        changed.target[:] = 99999
        a, b = enhance_windows(first, cfg), enhance_windows(changed, cfg)
        np.testing.assert_array_equal(a.gru_history, b.gru_history)
        np.testing.assert_array_equal(a.future_features, b.future_features)
        altered = frame.copy()
        altered.loc[288:, "pv_power_kw"] = 99999
        altered.loc[288:, "SWGDN"] = 9999
        other = load_and_regularize(altered, spec)
        w = build_windows(other, 288, 384, latitude=30.9, longitude=111.35,
                          history_weather_columns=spec.history_weather_columns)
        np.testing.assert_array_equal(a.gru_history[0], enhance_windows(w, cfg).gru_history[0])

    def test_diagnostic_lead_peak_shape_ramp_and_batch_statistics(self):
        day = np.ones((2, 96), dtype=bool)
        truth = np.tile(np.arange(96, dtype=float), (2, 1))
        pred = truth+2
        weather = np.full(truth.shape, "clear", dtype=object)
        weather[:, 48:] = "rain"
        meter = ForecastDiagnostics()
        meter.add(truth, pred, day, weather, np.array(["clear", "clear"]))
        value = meter.result()
        self.assertEqual(sum(m["count"] for m in value["daylight_by_lead"].values()), 192)
        self.assertEqual(value["daylight"]["rmse"], 2)
        self.assertEqual(value["peaks"]["peak_mae_kw"], 2)
        self.assertEqual(value["daylight_ramps_kw_per_15min"]["rmse"], 0)
        self.assertEqual(value["weather_transition_points"]["count"], 2)
        batched = ForecastDiagnostics()
        for i in range(2):
            batched.add(truth[i:i+1], pred[i:i+1], day[i:i+1], weather[i:i+1], np.array(["clear"]))
        self.assertEqual(value, batched.result())

    def test_tree_reload_public_prediction_and_missing_file_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, frame, spec, regular, windows, meta, model = tree_bundle(root)
            record = predict_record(root, frame)
            self.assertEqual(record["status"], "available")
            self.assertTrue(record["confidence"]["physics_features"])
            start = pd.Timestamp(record["forecast_start_at"])
            from pv_forecast.data import feature_frame, WindowedData
            cov, names = feature_frame(pd.DataFrame(index=pd.date_range(start, periods=96, freq="15min")), latitude=30.9, longitude=111.35)
            past = regular.frame[meta["raw_history_feature_names"]].iloc[-288:].to_numpy()
            w = WindowedData(past[:, :1][None], cov.to_numpy()[None], np.zeros((1, 96)), [start], names,
                             past[None], meta["raw_history_feature_names"])
            expected = _predict_windows(w, meta, model, None)[0]
            np.testing.assert_allclose(expected, [v["value"] for v in record["forecast_values"]])
            (root/"physics_tree.joblib").unlink()
            fallback = predict_record(root, frame)
            self.assertEqual(fallback["status"], "degraded")
            self.assertFalse(fallback["confidence"]["physics_features"])
            self.assertEqual(fallback["prediction_interval"]["method"], meta["baseline_fallback"]["calibration"]["method"])

    def test_oracle_separation_and_main_model_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, frame, spec, regular, windows, meta, model = tree_bundle(root)
            ordinary = backtest_bundles([root], frame, batch_windows=512)
            diagnostic = backtest_bundles([root], frame, batch_windows=512, oracle_diagnostic_config=cfg)
            self.assertEqual(ordinary["models"], diagnostic["models"])
            oracle = diagnostic["ideal_future_weather_diagnostic"]
            self.assertTrue(oracle["diagnostic_only"])
            self.assertFalse(oracle["model_selection_eligible"])
            self.assertFalse(oracle["production_eligible"])
            self.assertEqual(oracle["evaluated_windows"], diagnostic["evaluated_windows"])
            meta["oracle_future_weather"] = True
            (root/"model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "diagnostic"):
                predict_record(root, frame)

    def test_physical_tree_recalibration_preserves_model_and_saved_calibration_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, frame, spec, regular, windows, meta, model = tree_bundle(root/"parent")
            meta["data_splits"]["calibration"] = copy.deepcopy(meta["data_splits"]["validation"])
            (root/"parent"/"model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            before = predict_record(root/"parent", frame)
            result = recalibrate_bundle(root/"parent", frame, root/"derived")
            after = predict_record(root/"derived", frame)
            self.assertEqual(before["forecast_values"], after["forecast_values"])
            self.assertTrue((root/"derived"/"physics_tree.joblib").exists())
            self.assertEqual(result["report"]["calibration_partition"], meta["data_splits"]["calibration"])

    def test_enhanced_gru_power_units_and_save_reload(self):
        cfg, frame, spec, regular, windows = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = train_model(frame, root, spec, TrainingConfig(skip_gru=True))
            meta = result["metadata"]
            end = meta["data_splits"]["train"]["rows"]
            val_end = end+meta["data_splits"]["validation"]["rows"]
            train, val = enhance_windows(windows(0, end), cfg), enhance_windows(windows(end, val_end), cfg)
            target, future = _fit_scalers(train)
            scalers = {"target_scaler": target, "feature_scaler": future, "history_scaler": _fit_history_scaler(train)}
            st, sv = _scale_windows(train, **scalers), _scale_windows(val, **scalers)
            gru = GRUForecaster(config=GRUConfig(units=4, epochs=1, batch_size=128, patience=1))
            gru.fit(st.gru_history, st.future_features, st.target, validation_data=(sv.gru_history, sv.future_features, sv.target),
                     target_scaler=target, output_dir=root)
            gru.save(str(root/"gru.keras"))
            joblib.dump(scalers, root/"scalers.joblib")
            meta.update(selected_model="gru", point_correction=None, physics_config=cfg,
                raw_feature_names=meta["feature_names"], raw_history_feature_names=meta["history_feature_names"],
                feature_names=val.feature_names, history_feature_names=val.history_feature_names)
            (root/"model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            expected = _predict_windows(windows(end, val_end), meta, gru, scalers)
            reloaded = GRUForecaster.load(str(root/"gru.keras"))
            actual = _predict_windows(windows(end, val_end), meta, reloaded, scalers)
            np.testing.assert_allclose(expected, actual, atol=.001, rtol=0)
            record = predict_record(root, frame)
            self.assertEqual(record["status"], "available")
            self.assertTrue(all(0 <= p["value"] <= 5000 for p in record["forecast_values"]))
            self.assertEqual(gru.training_summary["metric_units"]["rmse_kw"], "kW")
            self.assertEqual(gru.training_summary["training_windows_used"], train.size)

    def test_cli_diagnostic_and_comparison_interfaces(self):
        parser = build_parser()
        args = parser.parse_args(["diagnose", "--input", "x.csv", "--bundle-dir", "a", "b", "--output", "diagnosis.json", "--oracle-physics-config", "config.json"])
        self.assertEqual(args.oracle_physics_config, "config.json")
        args = parser.parse_args(["compare-physics", "--input", "x.csv", "--control-bundle", "old", "--physics-config", "config.json", "--output-dir", "new"])
        self.assertEqual(args.batch_windows, 1024)


if __name__ == "__main__":
    unittest.main()
