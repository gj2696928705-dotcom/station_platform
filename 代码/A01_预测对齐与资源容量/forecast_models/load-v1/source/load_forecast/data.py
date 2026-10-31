"""Strict interval-average input and causal, direct multi-horizon features."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INTERVAL = pd.Timedelta(minutes=15)
HORIZON = 96
HISTORY = 672
TARGETS = {"base_load_kw": "base_load_excluding_charging_pv_storage",
           "total_load_kw": "gross_load_including_charging_excluding_pv_storage"}


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class InputSpec:
    timestamp_col: str = "timestamp"
    target_col: str = "base_load_kw"
    timezone: str = "Asia/Shanghai"
    station_id: str = "HH-Huanghua-CSS"
    input_kind: str = "interval_average_power"
    energy_interval_minutes: int | None = None
    source_target_col: str | None = None

    def __post_init__(self):
        if self.target_col not in TARGETS:
            raise ValueError(f"target_col must be one of {list(TARGETS)}")
        if self.input_kind not in {"interval_average_power", "interval_energy"}:
            raise ValueError("Only interval average kW or explicitly mapped interval kWh is supported")
        if self.input_kind == "interval_energy" and (
                self.energy_interval_minutes != 15 or not self.source_target_col):
            raise ValueError("interval_energy requires energy_interval_minutes=15 and source_target_col")
        if not self.station_id:
            raise ValueError("station_id must be explicit")


@dataclass
class LoadData:
    frame: pd.DataFrame
    quality: dict
    spec: InputSpec


def timestamp(value, timezone):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise DataValidationError("invalid timestamp")
    return result.tz_localize(timezone) if result.tzinfo is None else result.tz_convert(timezone)


def load_data(source, spec=InputSpec()):
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"input file not found: {path}")
        if path.suffix.lower() == ".csv":
            raw = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            raw = pd.read_excel(path)
        else:
            raise DataValidationError("input must be CSV or Excel")
    source_col = spec.source_target_col or spec.target_col
    required = {spec.timestamp_col, source_col, "station_id", "is_simulated"}
    if required - set(raw):
        raise DataValidationError(f"missing columns: {sorted(required - set(raw))}")
    if raw.empty:
        raise DataValidationError("empty input")
    if not raw.station_id.eq(spec.station_id).all():
        raise DataValidationError("station_id missing, mixed or different from configured station")
    markers = raw.is_simulated.astype(str).str.lower()
    if not markers.isin(["true", "false"]).all() or markers.nunique() != 1:
        raise DataValidationError("is_simulated must consistently be true or false")
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(raw[spec.timestamp_col], errors="raise"))
        idx = idx.tz_localize(spec.timezone) if idx.tz is None else idx.tz_convert(spec.timezone)
    except Exception as exc:
        raise DataValidationError(f"invalid timestamps: {exc}") from exc
    if idx.hasnans or idx.has_duplicates or not idx.is_monotonic_increasing:
        raise DataValidationError("timestamps must be valid, unique and strictly increasing")
    if not (idx == idx.floor("15min")).all():
        raise DataValidationError("timestamps must lie on the 15-minute grid; aggregation is not implicit")
    values = pd.to_numeric(raw[source_col], errors="coerce")
    if (raw[source_col].notna() & values.isna()).any():
        raise DataValidationError("target contains non-numeric values")
    if np.isinf(values.to_numpy()).any() or (values.dropna() < 0).any():
        raise DataValidationError("load must be finite and nonnegative (NaN remains a missing observation)")
    components = ["base_load_kw", "charging_load_kw", "total_load_kw"]
    if set(components).issubset(raw.columns) and spec.input_kind == "interval_average_power":
        parts = raw[components].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        complete = np.isfinite(parts).all(axis=1)
        if np.isinf(parts).any() or (parts[np.isfinite(parts)] < 0).any():
            raise DataValidationError("load components must be finite and nonnegative")
        if not np.allclose(parts[complete, 0] + parts[complete, 1], parts[complete, 2], rtol=1e-8, atol=2e-6):
            raise DataValidationError("total_load_kw must equal base_load_kw + charging_load_kw")
    if spec.input_kind == "interval_energy":
        values = values / .25
    # No interpolation or backfill: a later value must never repair an earlier origin.
    grid = pd.date_range(idx[0], idx[-1], freq="15min")
    if len(grid) > 2_000_000:
        raise DataValidationError("input span exceeds supported offline dataset size")
    frame = pd.DataFrame({spec.target_col: values.to_numpy(float)}, index=idx).reindex(grid)
    quality = {"source_rows": len(raw), "regular_rows": len(frame),
               "missing_rows": int(frame[spec.target_col].isna().sum()),
               "inserted_timestamps": len(grid) - len(idx), "imputed_rows": 0,
               "source_kind": "simulated" if markers.iloc[0] == "true" else "measured_unverified",
               "station_id": spec.station_id, "timezone": spec.timezone,
               "interval_minutes": 15, "power_semantics": "interval_average_kw",
               "first_timestamp": grid[0].isoformat(), "last_timestamp": grid[-1].isoformat()}
    return LoadData(frame, quality, spec)


def valid_origins(values, start, stop, stride=4):
    """All labels lie in [start, stop); all historical inputs precede origin."""
    first = max(start, HISTORY)
    first += (-first) % stride
    origins = np.arange(first, stop - HORIZON + 1, stride, dtype=np.int64)
    missing = np.r_[0, np.cumsum(~np.isfinite(values))]
    return origins[(missing[origins + HORIZON] - missing[origins - HISTORY]) == 0]


def features(values, index, origins):
    """One shared XGBoost model: (origin, lead) -> power, without recursion."""
    origins = np.asarray(origins, dtype=np.int64)
    n = len(origins)
    if not n or origins.min() < HISTORY or origins.max() > len(values):
        raise DataValidationError("insufficient history to construct features")
    lead = np.arange(HORIZON)
    pos = origins[:, None] + lead[None, :]
    # Periodic references are ALWAYS before the forecast origin, including lead 95.
    previous_days = values[pos[:, :, None] - np.arange(1, 8)[None, None, :] * HORIZON]
    cols, names = [], []

    def add(name, arr):
        cols.append(np.broadcast_to(arr, (n, HORIZON)).reshape(-1))
        names.append(name)

    add("lead_step", lead[None, :])
    add("previous_day_same_slot", previous_days[:, :, 0])
    add("previous_week_same_slot", previous_days[:, :, -1])
    add("previous_7days_slot_mean", previous_days.mean(axis=2))
    add("previous_7days_slot_std", previous_days.std(axis=2))
    for lag in (1, 2, 4, 96, 672):
        add(f"origin_lag_{lag}", values[origins - lag, None])
    for window in (4, 96, 672):
        history = values[origins[:, None] - np.arange(1, window + 1)[None, :]]
        for stat in ("mean", "std", "min", "max"):
            add(f"history_{window}_{stat}", getattr(history, stat)(axis=1)[:, None])
    target_times = pd.DatetimeIndex(index[0] + pd.to_timedelta(pos.reshape(-1) * 15, unit="min"))
    slots = np.asarray(target_times.hour * 4 + target_times.minute // 15).reshape(n, HORIZON)
    dow = np.asarray(target_times.dayofweek).reshape(n, HORIZON)
    doy = np.asarray(target_times.dayofyear).reshape(n, HORIZON)
    for label, number, period in (("slot", slots, 96), ("weekday", dow, 7), ("year", doy - 1, 365.25)):
        add(f"{label}_sin", np.sin(2 * np.pi * number / period))
        add(f"{label}_cos", np.cos(2 * np.pi * number / period))
    add("weekday", dow)
    add("is_weekend", (dow >= 5).astype(float))
    result = np.column_stack(cols).astype(np.float32)
    if not np.isfinite(result).all():
        raise DataValidationError("feature history has missing or non-finite values")
    return result, names


def labels(values, origins):
    return values[origins[:, None] + np.arange(HORIZON)[None, :]]


def baseline_predictions(values, origins):
    pos = origins[:, None] + np.arange(HORIZON)[None, :]
    return {
        "persistence": np.repeat(values[origins - 1, None], HORIZON, axis=1),
        "previous_day": values[pos - HORIZON],
        "previous_week": values[pos - HISTORY],
        "same_slot_mean": values[pos[:, :, None] - np.arange(1, 8)[None, None, :] * HORIZON].mean(axis=2),
    }
