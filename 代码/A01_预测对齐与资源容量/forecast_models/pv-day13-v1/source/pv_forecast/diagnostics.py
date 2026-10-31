"""Point-forecast diagnostics by lead, weather, peaks, shape and ramps."""
from __future__ import annotations

import numpy as np

from .experiments import MetricAccumulator

LEAD_HOURS = ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24))


class PointStats:
    def __init__(self):
        self.meter = MetricAccumulator()
        self.error_sum = 0.

    def add(self, truth, prediction, mask=None):
        a, b = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
        if mask is not None:
            a, b = a[mask], b[mask]
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("diagnostic truth and prediction must be aligned and finite")
        self.meter.add(a, b)
        self.error_sum += float(np.sum(b-a))

    def result(self):
        values = self.meter.result()
        values["mean_error_kw"] = self.error_sum/values["count"] if values["count"] else None
        return values


class ForecastDiagnostics:
    def __init__(self, interval_minutes=15):
        if interval_minutes != 15:
            raise ValueError("diagnostic lead bins currently require 15-minute forecasts")
        self.interval_minutes = interval_minutes
        self.overall = PointStats()
        self.leads = {f"{lo:02d}_{hi:02d}h": PointStats() for lo, hi in LEAD_HOURS}
        self.weather = {key: {} for key in self.leads}
        self.ramps, self.transition_points, self.transition_ramps = PointStats(), PointStats(), PointStats()
        self.peak_error, self.peak_abs, self.peak_n = 0., 0., 0
        self.timing_abs, self.timing_n = 0., 0
        self.shape_sse, self.shape_n = 0., 0

    def add(self, truth, prediction, daylight, weather=None, preceding_weather=None):
        a, b, day = [np.asarray(x) for x in (truth, prediction, daylight)]
        if a.ndim != 2 or a.shape != b.shape or a.shape != day.shape or a.shape[1] != 96 or day.dtype != bool:
            raise ValueError("diagnostics require aligned 96-step truth, forecast and boolean daylight")
        self.overall.add(a, b, day)
        for lo, hi in LEAD_HOURS:
            key = f"{lo:02d}_{hi:02d}h"
            mask = day.copy()
            mask[:, :lo*4] = False
            mask[:, hi*4:] = False
            self.leads[key].add(a, b, mask)
            if weather is not None:
                for state in np.unique(np.asarray(weather).astype(str)):
                    if state not in ("nan", "None", "<NA>"):
                        self.weather[key].setdefault(state, PointStats()).add(a, b, mask & (weather == state))
        da, db = np.diff(a, axis=1), np.diff(b, axis=1)
        pair_day = day[:, 1:] & day[:, :-1]
        self.ramps.add(da, db, pair_day)
        if weather is not None:
            changes = np.zeros_like(day)
            labels = np.asarray(weather)
            changes[:, 1:] = labels[:, 1:] != labels[:, :-1]
            if preceding_weather is not None:
                changes[:, 0] = labels[:, 0] != np.asarray(preceding_weather)
            known = ~np.isin(labels.astype(str), ["nan", "None", "<NA>"])
            changes &= known
            self.transition_points.add(a, b, day & changes)
            self.transition_ramps.add(da, db, pair_day & changes[:, 1:])
        true_day = np.where(day, a, -np.inf)
        pred_day = np.where(day, b, -np.inf)
        usable = day.any(axis=1)
        peaks_a, peaks_b = np.max(true_day, axis=1)[usable], np.max(pred_day, axis=1)[usable]
        error = peaks_b-peaks_a
        self.peak_error += float(error.sum())
        self.peak_abs += float(np.abs(error).sum())
        self.peak_n += len(error)
        positive = usable & (np.max(true_day, axis=1) > 1.) & (np.max(pred_day, axis=1) > 1.)
        if positive.any():
            idx_a, idx_b = np.argmax(true_day[positive], axis=1), np.argmax(pred_day[positive], axis=1)
            self.timing_abs += float(np.abs(idx_b-idx_a).sum())*self.interval_minutes
            self.timing_n += int(positive.sum())
            shape_a = a[positive]/np.max(true_day[positive], axis=1)[:, None]
            shape_b = b[positive]/np.max(pred_day[positive], axis=1)[:, None]
            delta = (shape_b-shape_a)[day[positive]]
            self.shape_sse += float(np.sum(delta**2))
            self.shape_n += delta.size

    def result(self):
        return {"daylight": self.overall.result(),
            "daylight_by_lead": {key: value.result() for key, value in self.leads.items()},
            "daylight_weather_by_lead": {key: {state: value.result() for state, value in values.items()}
                                        for key, values in self.weather.items()},
            "daylight_ramps_kw_per_15min": self.ramps.result(),
            "weather_transition_points": self.transition_points.result(),
            "weather_transition_ramps_kw_per_15min": self.transition_ramps.result(),
            "peaks": {"windows": self.peak_n, "mean_peak_error_kw": self.peak_error/self.peak_n if self.peak_n else None,
                      "peak_mae_kw": self.peak_abs/self.peak_n if self.peak_n else None,
                      "timing_windows": self.timing_n,
                      "timing_mae_minutes": self.timing_abs/self.timing_n if self.timing_n else None},
            "normalized_daylight_shape": {"points": self.shape_n, "rmse": float(np.sqrt(self.shape_sse/self.shape_n)) if self.shape_n else None},
            "definitions": {"leads": "half-open hours from origin: [0,1),[1,3),[3,6),[6,12),[12,24)",
                "bias": "prediction minus observed power",
                "shape": "each daylight curve divided by its own peak; requires both peaks >1 kW",
                "transitions": "observed weather_type changes used only after prediction; no forecast input",
                "overlap": "points and ramps count separately for every forecast origin"}}
