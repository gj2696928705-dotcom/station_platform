"""Deterministic clear-sky PV geometry and causal, historical physical features."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import WindowedData

HISTORY_EXTRA = ("clear_pv_kw", "clear_ghi_w_m2", "power_clear_ratio", "ghi_clear_ratio",
                 "power_ratio_delta", "cloud_delta", "cloud_squared", "power_ratio_mean_1h", "cloud_mean_1h")
FUTURE_EXTRA = ("clear_pv_kw", "clear_ghi_w_m2")


def load_physics_config(source):
    """Use explicitly configured simulated plant parameters, never fit on test."""
    config = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else copy.deepcopy(source)
    if "station" in config:
        config = {"station": config["station"], "pv": config["pv"],
            "clear_ghi_coefficient": config["weather"]["haurwitz_coefficient_w_m2"],
            "clear_ghi_attenuation": config["weather"]["haurwitz_attenuation"],
            "clear_diffuse_fraction": config["weather"]["clear_diffuse_fraction"],
            "ambient_reference_c": 20., "power_ratio_floor_fraction": .01, "ghi_ratio_floor_w_m2": 20.,
            "parameter_basis": "configured_simulated_plant; not real-site verified"} if "weather" in config else config
    station, pv = config["station"], config["pv"]
    required = ("latitude_deg", "longitude_deg", "capacity_kw")
    if any(not np.isfinite(station[k]) for k in required) or station["capacity_kw"] <= 0:
        raise ValueError("finite positive PV capacity and explicit coordinates required")
    if not (-90 <= station["latitude_deg"] <= 90 and -180 <= station["longitude_deg"] <= 180):
        raise ValueError("invalid physical coordinates")
    for key in ("tilt_deg", "azimuth_deg", "albedo", "noct_c", "temperature_coefficient_per_c",
                "reference_cell_temperature_c", "reference_irradiance_w_m2", "system_loss_fraction"):
        if not np.isfinite(pv[key]):
            raise ValueError("physical PV parameters must be finite")
    if (pv["reference_irradiance_w_m2"] <= 0 or not 0 <= pv["system_loss_fraction"] < 1
            or not 0 <= pv["albedo"] <= 1 or not 0 <= pv["tilt_deg"] <= 90
            or config["clear_ghi_coefficient"] <= 0 or config["clear_ghi_attenuation"] < 0
            or not 0 <= config["clear_diffuse_fraction"] <= 1
            or config["power_ratio_floor_fraction"] <= 0 or config["ghi_ratio_floor_w_m2"] <= 0):
        raise ValueError("invalid clear-sky or normalization parameters")
    return config


def solar_vectors(index, latitude, longitude):
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise ValueError("physical solar geometry requires timezone-aware timestamps")
    # UTC fractional-year approximation: a display timezone must not change
    # physical geometry. This is independent of the simulator's implementation.
    index = index.tz_convert("UTC")
    hours = np.asarray(index.hour+index.minute/60+index.second/3600, dtype=float)
    gamma = 2*np.pi/365*(index.dayofyear.to_numpy()-1+(hours-12)/24)
    eq = 229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)
                 -.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    dec = (.006918-.399912*np.cos(gamma)+.070257*np.sin(gamma)-.006758*np.cos(2*gamma)
           +.000907*np.sin(2*gamma)-.002697*np.cos(3*gamma)+.00148*np.sin(3*gamma))
    angle = np.deg2rad((hours*60+eq+4*longitude) % 1440/4-180)
    lat = np.deg2rad(latitude)
    east = -np.cos(dec)*np.sin(angle)
    north = np.cos(lat)*np.sin(dec)-np.sin(lat)*np.cos(dec)*np.cos(angle)
    up = np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(angle)
    return east, north, up


def irradiance_to_power(index, ghi, dhi, dni, temperature, config):
    """Physical kW calculation; weather arguments are explicit, not fetched."""
    s, p = config["station"], config["pv"]
    east, north, up = solar_vectors(index, s["latitude_deg"], s["longitude_deg"])
    ghi, dhi, dni, temperature = [np.asarray(x, dtype=float) for x in (ghi, dhi, dni, temperature)]
    if any(x.shape != up.shape or not np.isfinite(x).all() for x in (ghi, dhi, dni, temperature)):
        raise ValueError("physical weather arrays must be finite and match timestamps")
    if any((x < 0).any() for x in (ghi, dhi, dni)):
        raise ValueError("irradiance cannot be negative")
    tilt, azimuth = np.deg2rad([p["tilt_deg"], p["azimuth_deg"]])
    incidence = up*np.cos(tilt)+np.sin(tilt)*(north*np.cos(azimuth)+east*np.sin(azimuth))
    poa = dni*np.maximum(incidence, 0)+dhi*(1+np.cos(tilt))/2+ghi*p["albedo"]*(1-np.cos(tilt))/2
    cell = temperature+(p["noct_c"]-20)/800*poa
    factor = np.maximum(0, 1+p["temperature_coefficient_per_c"]*(cell-p["reference_cell_temperature_c"]))
    power = s["capacity_kw"]*poa/p["reference_irradiance_w_m2"]*factor*(1-p["system_loss_fraction"])
    return np.where(up > 1e-6, np.clip(power, 0, s["capacity_kw"]), 0)


def clear_profile(index, config):
    s = config["station"]
    _, _, up = solar_vectors(index, s["latitude_deg"], s["longitude_deg"])
    ghi = np.zeros(len(up))
    day = up > 1e-6
    ghi[day] = config["clear_ghi_coefficient"]*up[day]*np.exp(-config["clear_ghi_attenuation"]/up[day])
    dhi = ghi*config["clear_diffuse_fraction"]
    dni = np.divide(ghi-dhi, up, out=np.zeros_like(ghi), where=day)
    power = irradiance_to_power(index, ghi, dhi, dni, np.full(len(up), config["ambient_reference_c"]), config)
    return power, ghi


def _rolling_mean(x, size):
    sums = np.cumsum(x, axis=1, dtype=float)
    prior = np.concatenate([np.zeros((len(x), size)), sums[:, :-size]], axis=1) if x.shape[1] > size else np.zeros_like(sums)
    return (sums-prior)/np.minimum(np.arange(1, x.shape[1]+1), size)[None]


def enhance_windows(windows, config):
    """Transform only historical observations and deterministic future geometry.

    Targets, diagnostic weather labels and unissued future observations are
    never read. Entire windows stay present; the ratio floor applies at dawn.
    """
    if "clear_pv_kw" in windows.feature_names:
        if tuple(windows.feature_names[-len(FUTURE_EXTRA):]) != FUTURE_EXTRA or tuple(windows.history_feature_names[-len(HISTORY_EXTRA):]) != HISTORY_EXTRA:
            raise ValueError("invalid enhanced physical feature order")
        return windows
    if windows.size == 0:
        raise ValueError("physical enhancement requires nonempty windows")
    hist = np.asarray(windows.gru_history, dtype=np.float32)
    names = list(windows.history_feature_names)
    if not {"SWGDN", "cloud_cover"}.issubset(names) or windows.target.shape[1] != 96 or hist.shape[1] != 288:
        raise ValueError("physics features require 288 history/96 target points, historical SWGDN and cloud_cover")
    starts = pd.DatetimeIndex(windows.forecast_starts)
    step = 15*60*10**9
    lo = starts.min()-pd.Timedelta(minutes=hist.shape[1]*15)
    hi = starts.max()+pd.Timedelta(minutes=(windows.target.shape[1]-1)*15)
    grid = pd.date_range(lo, hi, freq="15min")
    positions = (starts.asi8-lo.value)//step
    if not np.array_equal(lo.value+positions*step, starts.asi8):
        raise ValueError("forecast starts must lie on a 15-minute physical grid")
    hp, hg = clear_profile(grid, config)
    ph = positions[:, None]-hist.shape[1]+np.arange(hist.shape[1])[None]
    pf = positions[:, None]+np.arange(windows.target.shape[1])[None]
    clear_h, ghi_h = hp[ph], hg[ph]
    power_ratio = hist[..., 0]/np.maximum(clear_h, config["station"]["capacity_kw"]*config["power_ratio_floor_fraction"])
    ghi_ratio = hist[..., names.index("SWGDN")]/np.maximum(ghi_h, config["ghi_ratio_floor_w_m2"])
    cloud = hist[..., names.index("cloud_cover")]
    delta_p = np.diff(power_ratio, axis=1, prepend=power_ratio[:, :1])
    delta_c = np.diff(cloud, axis=1, prepend=cloud[:, :1])
    extra = np.stack([clear_h, ghi_h, power_ratio, ghi_ratio, delta_p, delta_c, cloud**2,
                      _rolling_mean(power_ratio, 4), _rolling_mean(cloud, 4)], axis=-1).astype(np.float32)
    future = np.concatenate([np.asarray(windows.future_features, dtype=np.float32),
                             np.stack([hp[pf], hg[pf]], axis=-1).astype(np.float32)], axis=-1)
    enhanced_history = np.concatenate([hist, extra], axis=-1)
    if not np.isfinite(future).all() or not np.isfinite(enhanced_history).all():
        raise ValueError("physical features must be finite")
    return WindowedData(windows.history_power, future, windows.target, windows.forecast_starts,
        [*windows.feature_names, *FUTURE_EXTRA], enhanced_history, [*names, *HISTORY_EXTRA], windows.weather_issued_at)


def clear_persistence(windows, config):
    enhanced = enhance_windows(windows, config)
    names = list(enhanced.history_feature_names)
    power = enhanced.gru_history[..., names.index("clear_pv_kw")]
    ratio = enhanced.gru_history[..., names.index("power_clear_ratio")]
    result = np.zeros(enhanced.target.shape)
    for row in range(enhanced.size):
        valid = np.flatnonzero(power[row] > config["station"]["capacity_kw"]*config["power_ratio_floor_fraction"])
        factor = float(np.mean(ratio[row, valid[-4:]])) if len(valid) else 0
        result[row] = enhanced.future_features[row, :, list(enhanced.feature_names).index("clear_pv_kw")]*max(factor, 0)
    return np.clip(result, 0, config["station"]["capacity_kw"])


class ClearPersistenceForecaster:
    def __init__(self, config, history_names, future_names):
        self.config, self.history_names, self.future_names = config, tuple(history_names), tuple(future_names)

    def predict(self, history, future):
        h, f = np.asarray(history), np.asarray(future)
        hn, fn = list(self.history_names), list(self.future_names)
        ratio, clear = h[..., hn.index("power_clear_ratio")], h[..., hn.index("clear_pv_kw")]
        factor = np.zeros(len(h))
        for row in range(len(h)):
            valid = np.flatnonzero(clear[row] > self.config["station"]["capacity_kw"]*self.config["power_ratio_floor_fraction"])
            factor[row] = np.mean(ratio[row, valid[-4:]]) if len(valid) else 0
        return np.clip(f[..., fn.index("clear_pv_kw")]*factor[:, None], 0, self.config["station"]["capacity_kw"])
