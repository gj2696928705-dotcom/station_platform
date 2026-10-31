"""Calendar-day samples: 39 historical power values and 13 future weather rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (DataValidationError, InputSpec, WindowedData, _read_source,
                   _apply_column_map, _normalise_timestamp)

DAY_LEN = 13
HISTORY_DAYS = 3
FUTURE_COLUMNS = ("SWGDN", "DHI", "DNI", "rain")
LAYOUT = "day13_07_19"


def day_index(day):
    day = pd.Timestamp(day)
    return pd.date_range(day.normalize() + pd.Timedelta(hours=7), periods=13, freq="h")


def forecast_start(value=None, timezone="Asia/Shanghai"):
    stamp = pd.Timestamp.now(tz=timezone) if value is None else pd.Timestamp(value)
    stamp = stamp.tz_localize(timezone) if stamp.tzinfo is None else stamp.tz_convert(timezone)
    if value is not None and (stamp.hour not in (0, 7) or stamp.minute or stamp.second or stamp.microsecond):
        raise DataValidationError("day13 forecast date must be a date or 07:00, never an intraday sliding origin")
    return day_index(stamp)[0]


def read_day_frame(source, spec=InputSpec()):
    if spec.input_kind != "power":
        raise DataValidationError("day13 inputs are power in kW; convert energy explicitly before use")
    frame = _apply_column_map(_read_source(source), spec)
    # The reference workbook calls these columns time and power.
    aliases = {}
    if spec.timestamp_col not in frame and "time" in frame:
        aliases["time"] = spec.timestamp_col
    if spec.target_col not in frame and "power" in frame:
        aliases["power"] = spec.target_col
    frame = frame.rename(columns=aliases)
    if spec.timestamp_col not in frame:
        raise DataValidationError(f"missing timestamp column: {spec.timestamp_col}")
    frame.index = _normalise_timestamp(frame[spec.timestamp_col], spec.timezone)
    frame = frame.drop(columns=[spec.timestamp_col]).sort_index()
    if frame.index.has_duplicates:
        raise DataValidationError("duplicate timestamps cannot form 13-row days")
    frame = frame.loc[(frame.index.hour >= 7) & (frame.index.hour <= 19)].copy()
    if len(frame) == 0:
        raise DataValidationError("no rows between 07:00 and 19:00")
    if (frame.index.minute != 0).any() or (frame.index.second != 0).any() or (frame.index.microsecond != 0).any():
        raise DataValidationError("day13 requires hourly rows; use prepare-day13 for sub-hourly data")
    return frame


def numeric_block(frame, index, columns):
    missing = set(columns).difference(frame.columns)
    if missing or not index.isin(frame.index).all():
        raise DataValidationError(f"incomplete 13-row day {index[0].date()}; missing columns {sorted(missing)}")
    value = frame.loc[index, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(value).all() or (value < 0).any():
        raise DataValidationError(f"{index[0].date()}: power/irradiance/rain must be finite and nonnegative")
    return value


def complete_days(frame):
    days = pd.DatetimeIndex(frame.index.normalize().unique())
    expected = pd.date_range(days[0], days[-1], freq="D")
    if not days.equals(expected):
        raise DataValidationError("missing calendar day; preceding three days must be consecutive")
    for day in days:
        actual = frame.index[frame.index.normalize() == day]
        if not actual.equals(day_index(day)):
            raise DataValidationError(f"{day.date()}: expected exactly 13 rows at 07:00–19:00; no trimming or padding")
    return days


def build_day_samples(frame, spec=InputSpec(), weather_archive=None):
    """One sample per target day. History can cross a split; targets cannot."""
    days = complete_days(frame)
    if len(days) < 4:
        raise DataValidationError("at least four complete days are required")
    numeric_block(frame, frame.index, [spec.target_col])
    histories, futures, targets, starts, issues = [], [], [], [], []
    for d in range(3, len(days)):
        target_index = day_index(days[d])
        hist_index = pd.DatetimeIndex(np.concatenate([day_index(day).to_numpy() for day in days[d-3:d]]))
        history = numeric_block(frame, hist_index, [spec.target_col])
        issue = None
        if weather_archive is None:
            features = numeric_block(frame, target_index, FUTURE_COLUMNS)
        else:
            weather, issue = weather_archive.at(target_index[0], target_index)
            if weather is None:
                raise DataValidationError(f"{days[d].date()}: no complete forecast issued by 07:00")
            features = numeric_block(weather, target_index, FUTURE_COLUMNS)
        histories.append(history)
        futures.append(features)
        targets.append(numeric_block(frame, target_index, [spec.target_col])[:, 0])
        starts.append(target_index[0])
        issues.append(issue)
    return WindowedData(np.asarray(histories), np.asarray(futures), np.asarray(targets), starts,
                        FUTURE_COLUMNS, np.asarray(histories), [spec.target_col], issues)


def subset(windows, indices):
    indices = np.asarray(indices, dtype=int)
    return WindowedData(windows.history_power[indices], windows.future_features[indices], windows.target[indices],
        [windows.forecast_starts[i] for i in indices], windows.feature_names,
        windows.history_features[indices], windows.history_feature_names,
        [windows.weather_issued_at[i] for i in indices])


def prepare_hourly(source, spec=InputSpec(), source_interval_minutes=15):
    """Explicit hourly mean power/irradiance; rain is interval accumulation."""
    if source_interval_minutes not in (15, 30, 60):
        raise DataValidationError("source interval must be 15, 30 or 60 minutes")
    frame = _apply_column_map(_read_source(source), spec)
    frame = frame.rename(columns={k: v for k, v in (("time", spec.timestamp_col), ("power", spec.target_col)) if v not in frame})
    frame.index = _normalise_timestamp(frame[spec.timestamp_col], spec.timezone)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise DataValidationError("duplicate timestamps")
    if not frame.index.equals(frame.index.floor(f"{source_interval_minutes}min")):
        raise DataValidationError("timestamps do not lie on the declared source interval grid")
    columns = [spec.target_col, *FUTURE_COLUMNS]
    values = numeric_block(frame, frame.index, columns)
    numeric = pd.DataFrame(values, index=frame.index, columns=columns)
    hourly = numeric.resample("h").agg({column: "sum" if column == "rain" else "mean" for column in columns})
    counts = numeric.resample("h").size()
    selected = (hourly.index.hour >= 7) & (hourly.index.hour <= 19)
    if (counts[selected] != 60 // source_interval_minutes).any():
        raise DataValidationError("incomplete source hour; hourly conversion does not fill missing rows")
    hourly = hourly.loc[selected].copy()
    complete_days(hourly)
    hourly.insert(0, "timestamp", hourly.index)
    return hourly.reset_index(drop=True)
