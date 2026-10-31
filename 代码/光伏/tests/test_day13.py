"""Daily reference contract, calendar boundaries, causal inputs and real GRU IO."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from pv_forecast.data import DataValidationError, InputSpec
from pv_forecast.day13_data import (LAYOUT, FUTURE_COLUMNS, build_day_samples, complete_days,
    day_index, forecast_start, prepare_hourly, read_day_frame)
from pv_forecast.day13 import Day13TrainingConfig, validate_spec
from pv_forecast.pipeline import train_model, predict_record, evaluate_bundle
from pv_forecast.records import validate_forecast_record
from pv_forecast.weather import WeatherForecastArchive


def daily_data(days=40):
    stamps = pd.date_range("2025-01-01", periods=days, freq="D", tz="Asia/Shanghai")
    index = day_index(stamps[0]).append([day_index(day) for day in stamps[1:]])
    sunlight = np.maximum(0., np.sin(np.pi*(index.hour.to_numpy()-6)/14))
    return pd.DataFrame({"timestamp": index, "pv_power_kw": sunlight*(600+np.repeat(np.arange(days)*3, 13)),
        "SWGDN": sunlight*500, "DHI": sunlight*100, "DNI": sunlight*400,
        "rain": np.repeat(np.arange(days)%3*.1, 13)})


class Day13DataTests(unittest.TestCase):
    def test_calendar_shapes_and_exact_previous_three_days(self):
        frame = daily_data(8)
        frame["pv_power_kw"] = np.arange(len(frame))
        windows = build_day_samples(read_day_frame(frame))
        self.assertEqual(windows.history_power.shape, (5, 39, 1))
        self.assertEqual(windows.future_features.shape, (5, 13, 4))
        self.assertEqual(windows.target.shape, (5, 13))
        np.testing.assert_array_equal(windows.history_power[0, :, 0], np.arange(39))
        np.testing.assert_array_equal(windows.history_power[1, :, 0], np.arange(13, 52))
        np.testing.assert_array_equal(windows.target[0], np.arange(39, 52))
        self.assertEqual(windows.feature_names, FUTURE_COLUMNS)
        self.assertEqual(windows.forecast_starts[0], pd.Timestamp("2025-01-04T07:00+08:00"))

    def test_incomplete_days_duplicates_and_calendar_gaps_rejected(self):
        original = daily_data(8)
        for frame in (original.iloc[:-1], original.drop(index=15), original.drop(index=range(26, 39)),
                      pd.concat([original, original.iloc[[0]]])):
            with self.subTest(rows=len(frame)), self.assertRaises(DataValidationError):
                build_day_samples(read_day_frame(frame))

    def test_hour_grid_no_fractional_sampling_or_silent_trimming(self):
        frame = daily_data(4)
        frame.loc[0, "timestamp"] += pd.Timedelta(minutes=15)
        with self.assertRaises(DataValidationError):
            read_day_frame(frame)
        with self.assertRaises(DataValidationError):
            build_day_samples(read_day_frame(daily_data(3)))
        with self.assertRaises(DataValidationError):
            forecast_start("2025-01-04T10:00+08:00")
        self.assertEqual(forecast_start("2025-01-04").hour, 7)
        self.assertEqual(forecast_start().date(), pd.Timestamp.now(tz="Asia/Shanghai").date())

    def test_only_power_in_history_and_no_target_in_future_branch(self):
        frame = daily_data(4)
        base = build_day_samples(read_day_frame(frame))
        changed = frame.copy()
        changed.loc[:38, list(FUTURE_COLUMNS)] = -999
        changed.loc[39:, "pv_power_kw"] *= 100
        windows = build_day_samples(read_day_frame(changed))
        np.testing.assert_array_equal(base.history_power, windows.history_power)
        np.testing.assert_array_equal(base.future_features, windows.future_features)
        self.assertFalse(np.array_equal(base.target, windows.target))

    def test_reference_column_aliases_and_feature_restrictions(self):
        frame = read_day_frame(daily_data(4).rename(columns={"timestamp": "time", "pv_power_kw": "power"}))
        self.assertEqual(build_day_samples(frame).size, 1)
        with self.assertRaises(DataValidationError):
            validate_spec(InputSpec(history_weather_columns=("temperature_c",)))
        with self.assertRaises(DataValidationError):
            validate_spec(InputSpec(weather_columns=("SWGDN", "rain")))

    def test_hourly_units_mean_power_and_accumulated_rain(self):
        index = pd.date_range("2025-01-01", periods=4*96, freq="15min", tz="Asia/Shanghai")
        frame = pd.DataFrame({"timestamp": index, "pv_power_kw": index.minute+100,
            "SWGDN": 200., "DHI": 100., "DNI": 100., "rain": .25})
        hourly = prepare_hourly(frame)
        self.assertEqual(len(hourly), 52)
        self.assertEqual(list(hourly.timestamp.dt.hour[:13]), list(range(7, 20)))
        np.testing.assert_array_equal(hourly.pv_power_kw, 122.5)
        np.testing.assert_array_equal(hourly.rain, 1.)
        with self.assertRaises(DataValidationError):
            prepare_hourly(frame.drop(index=30))


class Day13WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.bundle = Path(cls.scratch.name)/"gru"
        cls.source = daily_data()
        cls.result = train_model(cls.source, cls.bundle, config=Day13TrainingConfig(
            gru_epochs=2, gru_units=4, gru_batch_size=16, gru_patience=2))

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_real_gru_shapes_train_scalers_and_saved_best_epoch(self):
        from pv_forecast.models import GRUForecaster
        model = GRUForecaster.load(str(self.bundle/"gru.keras"))
        self.assertEqual(model.model.input_shape, [(None, 39, 1), (None, 13, 4)])
        self.assertEqual(model.model.output_shape, (None, 13, 1))
        meta = self.result["metadata"]
        self.assertEqual(meta["sampling_layout"], LAYOUT)
        self.assertEqual(meta["history_weather_columns"], [])
        self.assertEqual(meta["feature_names"], list(FUTURE_COLUMNS))
        count = meta["data_splits"]["train"]["samples"]
        self.assertEqual(count, 25)
        scalers = joblib.load(self.bundle/"scalers.joblib")
        self.assertEqual(scalers["history_scaler"].n_samples_seen_, count*39)
        self.assertEqual(scalers["feature_scaler"].n_samples_seen_, count*13)
        self.assertEqual(scalers["feature_scaler"].n_features_in_, 4)
        self.assertEqual(meta["data_splits"]["selection_validation"]["samples"], 3)
        self.assertEqual(meta["data_splits"]["interval_calibration"]["samples"], 3)
        history = json.loads((self.bundle/"training_history.json").read_text(encoding="utf-8"))
        summary = json.loads((self.bundle/"training_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["best_epoch"], min(history, key=lambda row: row["val_loss"])["epoch"])
        self.assertEqual(summary["training_windows_used"], count)

    def test_prediction_13_hours_physical_units_and_future_target_isolation(self):
        first = predict_record(self.bundle, self.source, forecast_start="2025-02-09")
        changed = self.source.copy()
        changed.loc[changed.timestamp.dt.date >= pd.Timestamp("2025-02-09").date(), "pv_power_kw"] = np.nan
        # Unused historical weather can be malformed without affecting input.
        changed.loc[changed.timestamp.dt.date < pd.Timestamp("2025-02-09").date(), "SWGDN"] = np.nan
        second = predict_record(self.bundle, changed, forecast_start="2025-02-09")
        self.assertEqual(first["status"], "available")
        self.assertEqual(first["forecast_values"], second["forecast_values"])
        self.assertEqual(len(first["forecast_values"]), 13)
        self.assertEqual(first["interval_minutes"], 60)
        self.assertEqual(pd.Timestamp(first["forecast_end_at"]).hour, 19)
        self.assertTrue(all(v["unit"] == "kW" and np.isfinite(v["value"]) and v["value"] >= 0 for v in first["forecast_values"]))
        with self.assertRaises(ValueError):
            validate_forecast_record({**first, "forecast_values": first["forecast_values"][:-1]})

    def test_external_future_weather_needs_no_target_power(self):
        frame = self.source
        target = frame.timestamp.dt.date == pd.Timestamp("2025-02-09").date()
        hist = frame.loc[~target, ["timestamp", "pv_power_kw"]]
        future = frame.loc[target, ["timestamp", *FUTURE_COLUMNS]]
        result = predict_record(self.bundle, hist, forecast_start="2025-02-09", future_weather_source=future)
        combined = predict_record(self.bundle, frame, forecast_start="2025-02-09")
        self.assertEqual(result["forecast_values"], combined["forecast_values"])

    def test_issued_forecast_rejects_later_vintage(self):
        target = self.source.loc[self.source.timestamp.dt.date == pd.Timestamp("2025-02-09").date()]
        earlier = target.drop(columns="pv_power_kw").assign(issued_at="2025-02-09T06:00+08:00")
        later = earlier.assign(issued_at="2025-02-09T08:00+08:00", SWGDN=99999.)
        result = predict_record(self.bundle, self.source, forecast_start="2025-02-09",
            future_weather_source=pd.concat([earlier, later]))
        expected = predict_record(self.bundle, self.source, forecast_start="2025-02-09", future_weather_source=earlier)
        self.assertEqual(result["forecast_values"], expected["forecast_values"])
        self.assertEqual(result["confidence"]["weather_forecast_issued_at"], "2025-02-09T06:00:00+08:00")
        no_early = predict_record(self.bundle, self.source, forecast_start="2025-02-09", future_weather_source=later)
        self.assertEqual(no_early["status"], "degraded")

    def test_missing_weather_or_model_uses_13_point_calibrated_baseline(self):
        result = predict_record(self.bundle, self.source.drop(columns="rain"), forecast_start="2025-02-09")
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(len(result["forecast_values"]), 13)
        weight = self.bundle/"gru.keras"
        temporary = self.bundle/"temporarily_absent.keras"
        weight.rename(temporary)
        try:
            fallback = predict_record(self.bundle, self.source, forecast_start="2025-02-09")
            self.assertEqual(result["forecast_values"], fallback["forecast_values"])
            self.assertEqual(result["prediction_interval"], fallback["prediction_interval"])
        finally:
            temporary.rename(weight)

    def test_missing_previous_calendar_day_is_unavailable(self):
        source = self.source.loc[self.source.timestamp.dt.date != pd.Timestamp("2025-02-07").date()]
        result = predict_record(self.bundle, source, forecast_start="2025-02-09")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["forecast_values"], [])

    def test_evaluation_scores_one_sample_per_day(self):
        result = evaluate_bundle(self.bundle, self.source)
        self.assertEqual(result["evaluated_days"], 37)
        self.assertEqual(result["metrics"]["count"], 37*13)
        self.assertEqual(len(result["per_hour"]), 13)
        self.assertEqual(len(result["per_day"]), 37)

    def test_cli_defaults_and_chronological_daily_backtest(self):
        from pv_forecast.cli import build_parser
        from pv_forecast.backtesting import backtest_bundles
        args = build_parser().parse_args(["train", "--input", "data.csv", "--output-dir", "bundle"])
        self.assertEqual(args.sampling_layout, LAYOUT)
        result = backtest_bundles([self.bundle], self.source)
        self.assertEqual(result["evaluated_windows"], 37)
        self.assertFalse(result["weights_retrained"])


if __name__ == "__main__":
    unittest.main()
