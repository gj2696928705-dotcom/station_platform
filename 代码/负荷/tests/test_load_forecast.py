"""Causality, holdout isolation, interval semantics and real XGBoost round trips."""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from load_forecast.data import (DataValidationError, HISTORY, HORIZON, InputSpec,
                                baseline_predictions, features, load_data, valid_origins)
from load_forecast.models import calibrate, interval, metrics
from load_forecast.pipeline import (TrainingConfig, dump, evaluate_bundle, predict_record,
                                    sha256, train_model, validate_record)


def fixture(days=80, constant=False):
    idx = pd.date_range("2025-01-01", periods=days * 96, freq="15min", tz="Asia/Shanghai")
    t = np.arange(len(idx))
    rng = np.random.default_rng(2025)
    power = np.full(len(idx), 80.) if constant else (
        90 + 25 * np.sin(2 * np.pi * t / 96) + 9 * (idx.dayofweek.to_numpy() < 5)
        + .007 * t + rng.normal(0, 2, len(t)))
    return pd.DataFrame({"timestamp": idx, "base_load_kw": power,
                         "station_id": "HH-Huanghua-CSS", "is_simulated": True})


class InputTests(unittest.TestCase):
    def test_missing_gap_is_never_interpolated(self):
        raw = fixture(10).drop(index=range(720, 740))
        data = load_data(raw)
        self.assertEqual(data.quality["missing_rows"], 20)
        self.assertEqual(data.quality["imputed_rows"], 0)
        origins = valid_origins(data.frame.base_load_kw.to_numpy(), 0, len(data.frame), 1)
        self.assertFalse(any((origins > 624) & (origins < 1412)))

    def test_invalid_input_cases(self):
        raw = fixture(8)
        bad = []
        duplicate = raw.copy(); duplicate.loc[1, "timestamp"] = duplicate.loc[0, "timestamp"]; bad.append(duplicate)
        offgrid = raw.copy(); offgrid.loc[0, "timestamp"] += pd.Timedelta(minutes=1); bad.append(offgrid)
        negative = raw.copy(); negative.loc[2, "base_load_kw"] = -1; bad.append(negative)
        inf = raw.copy(); inf.loc[2, "base_load_kw"] = np.inf; bad.append(inf)
        station = raw.copy(); station.loc[2, "station_id"] = "different"; bad.append(station)
        mixed = raw.copy(); mixed.loc[2, "is_simulated"] = False; bad.append(mixed)
        bad.extend([raw.iloc[::-1], raw.drop(columns="station_id"), raw.iloc[:0]])
        for frame in bad:
            with self.subTest(case=len(frame)), self.assertRaises(DataValidationError):
                load_data(frame)

    def test_explicit_energy_conversion(self):
        raw = fixture(8).rename(columns={"base_load_kw": "base_kwh"})
        expected = raw.base_kwh.to_numpy() * 4
        spec = InputSpec(input_kind="interval_energy", energy_interval_minutes=15, source_target_col="base_kwh")
        np.testing.assert_allclose(load_data(raw, spec).frame.base_load_kw, expected)
        with self.assertRaises(ValueError):
            InputSpec(input_kind="interval_energy")

    def test_timezone_conversion(self):
        raw = fixture(8)
        raw.timestamp = raw.timestamp.dt.tz_convert("UTC")
        self.assertEqual(load_data(raw).frame.index[0].isoformat(), "2025-01-01T00:00:00+08:00")

    def test_non_numeric_is_not_silently_missing(self):
        raw = fixture(8)
        raw["base_load_kw"] = raw.base_load_kw.astype(object)
        raw.loc[10, "base_load_kw"] = "bad"
        with self.assertRaises(DataValidationError):
            load_data(raw)

    def test_component_conservation(self):
        raw = fixture(8)
        raw["charging_load_kw"] = 200.
        raw["total_load_kw"] = raw.base_load_kw + 200.
        load_data(raw)
        raw.loc[50, "total_load_kw"] += 1.
        with self.assertRaises(DataValidationError):
            load_data(raw)


class FeatureTests(unittest.TestCase):
    def test_no_target_leakage_including_last_lead(self):
        data = load_data(fixture(12))
        values = data.frame.base_load_kw.to_numpy()
        origin = 800
        x, names = features(values, data.frame.index, [origin])
        altered = values.copy(); altered[origin:] = 999999
        after, _ = features(altered, data.frame.index, [origin])
        np.testing.assert_array_equal(x, after)
        self.assertAlmostEqual(float(x[-1, names.index("previous_day_same_slot")]), values[origin - 1], places=4)

    def test_targets_stay_wholly_inside_split(self):
        values = np.ones(6000)
        for start, stop in ((0, 3000), (3000, 4200), (4200, 6000)):
            origins = valid_origins(values, start, stop, 1)
            self.assertTrue((origins >= max(start, HISTORY)).all())
            self.assertTrue((origins + HORIZON <= stop).all())

    def test_baseline_alignment_for_non_midnight_origin(self):
        values = np.arange(1000, dtype=float)
        pred = baseline_predictions(values, np.array([800]))
        np.testing.assert_array_equal(pred["previous_day"][0], values[704:800])
        np.testing.assert_array_equal(pred["previous_week"][0], values[128:224])

    def test_constant_r2_and_calibration_order_statistic(self):
        self.assertIsNone(metrics(np.ones(5), np.ones(5))["r2"])
        truth = np.tile(np.arange(25)[:, None], (1, 96))
        cal = calibrate(truth, np.zeros_like(truth), .8)
        self.assertEqual(cal["quantiles_kw"][0], 20)
        lo, hi = interval(np.ones(96), cal)
        self.assertTrue((lo == 0).all()); self.assertTrue((hi == 21).all())


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        cls.raw = fixture()
        cls.bundle = cls.directory / "model"
        cls.config = TrainingConfig(rounds=25, patience=5, stride=24, max_depth=3, threads=2)
        cls.result = train_model(cls.raw, cls.bundle, config=cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_training_artifacts_and_reload(self):
        for name in ("xgboost.json", "training_history.csv", "training_summary.json", "metrics.json", "data_split.json"):
            self.assertTrue((self.bundle / name).is_file())
        record = predict_record(self.bundle, self.raw)
        validate_record(record)
        self.assertEqual(record["status"], "available", record["reason"])
        self.assertEqual(len(record["forecast_values"]), 96)
        self.assertEqual(record["target_semantics"], "base_load_excluding_charging_pv_storage")
        self.assertFalse(record["decision_eligible"])
        self.assertEqual(record["forecast_values"], predict_record(self.bundle, self.raw)["forecast_values"])

    def test_test_labels_do_not_affect_weights_or_calibration(self):
        mutated = self.raw.copy()
        cut = int(len(mutated) * (self.config.train_fraction + self.config.validation_fraction))
        mutated.loc[cut:, "base_load_kw"] *= 10
        second = self.directory / "mutated_test"
        result = train_model(mutated, second, config=self.config)
        self.assertEqual(sha256(self.bundle / "xgboost.json"), sha256(second / "xgboost.json"))
        self.assertEqual(self.result["metadata"]["calibration"], result["metadata"]["calibration"])
        self.assertEqual(self.result["metadata"]["selected_model"], result["metadata"]["selected_model"])

    def test_future_rows_cannot_change_prediction(self):
        start = self.raw.timestamp.iloc[-96]
        left = predict_record(self.bundle, self.raw, forecast_start=start)
        mutated = self.raw.copy(); mutated.loc[mutated.index[-96:], "base_load_kw"] += 100000
        right = predict_record(self.bundle, mutated, forecast_start=start)
        self.assertEqual(left["status"], "available")
        self.assertEqual(left["forecast_values"], right["forecast_values"])
        self.assertEqual(left["input_data_quality_summary"]["future_rows_excluded"], 96)

    def test_gap_stale_insufficient_and_wrong_scope_unavailable(self):
        missing = self.raw.copy(); missing.loc[len(missing) - 5, "base_load_kw"] = np.nan
        start = self.raw.timestamp.iloc[-1] + pd.Timedelta(minutes=30)
        records = [predict_record(self.bundle, missing), predict_record(self.bundle, self.raw.iloc[-100:]),
                   predict_record(self.bundle, self.raw, forecast_start=start),
                   predict_record(self.bundle, self.raw, input_spec=InputSpec(target_col="total_load_kw")),
                   predict_record(self.bundle, self.raw, forecast_start=self.raw.timestamp.iloc[800])]
        for record in records:
            self.assertEqual(record["status"], "unavailable")
            self.assertEqual(record["forecast_values"], [])
            self.assertTrue(record["reason"])

    def test_missing_bundle_returns_unavailable_record(self):
        record = predict_record(self.directory / "missing", self.raw, self.directory / "unavailable.json")
        self.assertEqual(record["status"], "unavailable")
        self.assertTrue((self.directory / "unavailable.json").exists())

    def test_as_of_uses_only_completed_intervals(self):
        clock = self.raw.timestamp.iloc[-96] + pd.Timedelta(minutes=7)
        record = predict_record(self.bundle, self.raw, as_of=clock)
        self.assertEqual(record["status"], "available", record["reason"])
        self.assertEqual(record["forecast_start_at"], clock.floor("15min").isoformat())

    def test_evaluate_and_image_failure_are_explicit(self):
        result = evaluate_bundle(self.bundle, self.raw, self.directory / "eval" / "evaluation.json")
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["metrics"]["point_count"], 96)
        with patch("load_forecast.pipeline.forecast_plot", side_effect=OSError("image failure")):
            record = predict_record(self.bundle, self.raw, self.directory / "pred" / "forecast.json")
        self.assertEqual(record["status"], "available")
        self.assertEqual(record["visualization_errors"], ["image failure"])

    def test_existing_results_are_not_overwritten(self):
        with self.assertRaises(ValueError):
            train_model(self.raw, self.bundle, config=self.config)

    def test_baseline_can_win_and_xgboost_still_saved(self):
        bundle = self.directory / "constant"
        result = train_model(fixture(constant=True), bundle, config=replace(self.config, rounds=8))
        self.assertEqual(result["metadata"]["selected_model"], "baseline:persistence")
        self.assertTrue((bundle / "xgboost.json").is_file())
        record = predict_record(bundle, fixture(constant=True))
        self.assertEqual(record["status"], "available", record["reason"])
        self.assertTrue(all(p["value"] == 80 for p in record["forecast_values"]))

    def test_calibration_identity_mismatch_rejected(self):
        bundle = self.directory / "mismatched"
        bundle.mkdir()
        meta = dict(self.result["metadata"])
        meta["calibration"] = {**meta["calibration"], "model": "wrong"}
        dump(bundle / "model_metadata.json", meta)
        self.assertEqual(predict_record(bundle, self.raw)["status"], "unavailable")

    def test_malformed_bundles_return_unavailable_without_crashing(self):
        variants = [None, [], {**self.result["metadata"], "input_spec": None}]
        for key, value in (("quantiles_kw", 1), ("quantiles_kw", [[1] * 96]),
                           ("quantiles_kw", [float("nan")] * 96), ("nominal_coverage", True)):
            meta = deepcopy(self.result["metadata"])
            meta["calibration"][key] = value
            variants.append(meta)
        for i, meta in enumerate(variants):
            bundle = self.directory / f"malformed-{i}"
            bundle.mkdir()
            # Deliberately bypass dump's allow_nan=False to exercise invalid external JSON.
            (bundle / "model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            record = predict_record(bundle, self.raw)
            self.assertEqual(record["status"], "unavailable")
            self.assertEqual(record["forecast_values"], [])
            self.assertTrue(record["reason"])

    def test_bundle_contract_changes_are_rejected(self):
        for key, value in (("interval_minutes", 30), ("history_steps", 96), ("horizon_steps", 24),
                           ("target_semantics", "gross_load_including_charging_excluding_pv_storage")):
            bundle = self.directory / f"bad-contract-{key}"
            bundle.mkdir()
            dump(bundle / "model_metadata.json", {**self.result["metadata"], key: value})
            self.assertEqual(predict_record(bundle, self.raw)["status"], "unavailable")

    def test_record_contract_rejects_bad_units_times_and_numbers(self):
        original = predict_record(self.bundle, self.raw)
        self.assertEqual(original["status"], "available")
        mutations = [
            lambda r: r.update(interval_minutes=30),
            lambda r: r.update(forecast_type="pv_power"),
            lambda r: r.update(target_semantics="unknown"),
            lambda r: r.update(asset_scope=None),
            lambda r: r.update(generated_at="2026-01-01T00:00:00"),
            lambda r: r["forecast_values"][0].update(unit="W"),
            lambda r: r["prediction_interval"]["values"][0].update(unit="kWh"),
            lambda r: r["forecast_values"][0].update(value=True),
            lambda r: r["forecast_values"][0].update(value=float("inf")),
            lambda r: r["forecast_values"][0].update(timestamp="2026-01-01T00:01:00+08:00"),
            lambda r: r["prediction_interval"].update(available=False),
            lambda r: r["prediction_interval"].update(nominal_coverage=2),
            lambda r: r.update(forecast_end_at=r["forecast_start_at"]),
        ]
        for i, mutate in enumerate(mutations):
            record = deepcopy(original)
            mutate(record)
            with self.subTest(case=i), self.assertRaises(ValueError):
                validate_record(record)

    def test_unavailable_records_must_have_empty_intervals_and_reason(self):
        original = predict_record(self.directory / "absent", self.raw)
        for key, value in (("reason", None), ("prediction_interval", {"available": False, "values": [1]})):
            record = deepcopy(original)
            record[key] = value
            with self.assertRaises(ValueError):
                validate_record(record)

    def test_xgboost_can_win_and_corrupt_weights_are_rejected(self):
        raw = fixture()
        t = np.arange(len(raw))
        raw["base_load_kw"] = (90 + 25 * np.sin(2 * np.pi * t / 96)
                               + 20 * raw.timestamp.dt.dayofweek.to_numpy()
                               + np.random.default_rng(125).normal(0, 8, len(raw)))
        bundle = self.directory / "calendar"
        result = train_model(raw, bundle, config=replace(self.config, rounds=150, learning_rate=.12, max_depth=4, patience=15))
        self.assertEqual(result["metadata"]["selected_model"], "xgboost")
        record = predict_record(bundle, raw)
        self.assertEqual(record["status"], "available", record["reason"])
        self.assertEqual(record["forecast_values"], predict_record(bundle, raw)["forecast_values"])
        unicode_bundle = self.directory / "中文模型目录"
        shutil.copytree(bundle, unicode_bundle)
        self.assertEqual(record["forecast_values"], predict_record(unicode_bundle, raw)["forecast_values"])
        with (bundle / "xgboost.json").open("ab") as stream:
            stream.write(b"corrupted")
        self.assertEqual(predict_record(bundle, raw)["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
