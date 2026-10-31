"""Reject inconsistent records before they can reach a project consumer."""
from copy import deepcopy
import unittest

from pv_forecast.records import build_forecast_record, validate_forecast_record


def record(daily=True):
    values = [100.]*(13 if daily else 4)
    result = build_forecast_record("contract-test", values, "2026-10-26T07:00+08:00", 60 if daily else 15,
        "synthetic-test", "pv-test", "candidate", None, {}, {}, {}, lower=[80.]*len(values), upper=[120.]*len(values))
    if daily:
        result["confidence"]["sampling_layout"] = "day13_07_19"
    return result


class RecordAlignmentTests(unittest.TestCase):
    def test_valid_daily_and_legacy_grids(self):
        for daily in (True, False):
            validate_forecast_record(record(daily))

    def test_end_time_matches_actual_last_point(self):
        for daily in (True, False):
            broken = record(daily)
            broken["forecast_end_at"] = "2026-10-26T23:00+08:00"
            with self.subTest(daily=daily), self.assertRaises(ValueError):
                validate_forecast_record(broken)

    def test_intervals_must_cover_every_point(self):
        broken = record()
        broken["prediction_interval"]["values"].pop()
        with self.assertRaises(ValueError):
            validate_forecast_record(broken)

    def test_wrong_interval_date_or_grid_is_rejected(self):
        for timestamp in ("2026-10-25T07:00+08:00", "2026-10-26T07:15+08:00"):
            broken = record()
            broken["prediction_interval"]["values"][0]["timestamp"] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                validate_forecast_record(broken)

    def test_nonfinite_and_reversed_values_rejected(self):
        cases = []
        for field in ("value", "lower", "upper"):
            broken = record()
            items = broken["forecast_values"] if field == "value" else broken["prediction_interval"]["values"]
            items[0][field] = float("nan")
            cases.append(broken)
        broken = record()
        broken["prediction_interval"]["values"][0].update(lower=121., upper=120.)
        cases.append(broken)
        for broken in cases:
            with self.assertRaises(ValueError):
                validate_forecast_record(broken)

    def test_unavailable_empty_record_has_zero_length_coverage(self):
        result = build_forecast_record("none", [], "2026-10-26T07:00+08:00", 60,
            "test", "pv", "candidate", None, {}, None, {}, status="unavailable")
        result["confidence"]["sampling_layout"] = "day13_07_19"
        validate_forecast_record(result)


if __name__ == "__main__":
    unittest.main()
