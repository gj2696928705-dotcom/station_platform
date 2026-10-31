"""Independent hand-computable kW, low-sun ratio and causal time checks."""
from pathlib import Path
import copy
import json
import os
import sys

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
import numpy as np
import pandas as pd
out = Path(__file__).resolve().parent
pv = out.parents[1]
sys.path.insert(0, str(pv))
from pv_forecast.data import WindowedData, feature_frame
from pv_forecast.physics import load_physics_config, solar_vectors, irradiance_to_power, clear_profile, enhance_windows

physics = load_physics_config(pv.parent/"数据"/"光伏数据"/"simulation_config.json")
flat = copy.deepcopy(physics)
flat["pv"]["tilt_deg"] = 0
day = pd.DatetimeIndex([pd.Timestamp("2025-06-05T12:00:00+08:00")])
up = solar_vectors(day, 30.9, 111.35)[2]
assert up[0] > 0
power = irradiance_to_power(day, [1000.], [200.], 800./up, [20.], flat)
# Horizontal POA = DNI*up + DHI = 1000 W/m²;
# cell = 20 + (45-20)/800*1000 = 51.25°C;
# power = 5000*(1-.004*(51.25-25))*(1-.12) = 3938 kW.
np.testing.assert_allclose(power, [3938.], atol=1e-9, rtol=0)
hotter = irradiance_to_power(day, [1000.], [200.], 800./up, [30.], flat)
np.testing.assert_allclose(hotter, [3762.], atol=1e-9, rtol=0)
clipped = irradiance_to_power(day, [2000.], [400.], 1600./up, [20.], flat)
np.testing.assert_array_equal(clipped, [5000.])
night = pd.DatetimeIndex([pd.Timestamp("2025-06-05T00:00:00+08:00")])
np.testing.assert_array_equal(irradiance_to_power(night, [0.], [0.], [0.], [20.], flat), [0.])
for bad in (-1., float("nan")):
    try:
        irradiance_to_power(day, [bad], [0.], [0.], [20.], flat)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid explicit irradiance was accepted")
start = night[0]
past = pd.date_range(start-pd.Timedelta(days=3), periods=288, freq="15min")
future = pd.date_range(start, periods=96, freq="15min")
features, feature_names = feature_frame(pd.DataFrame(index=future), latitude=30.9, longitude=111.35)
history = np.zeros((1, 288, 4), dtype=np.float32)
history[..., 0] = 7
history[..., 1] = 9
history[..., 2] = 20
history[..., 3] = .3
w = WindowedData(history[..., :1], features.to_numpy()[None], np.zeros((1, 96)), [start], feature_names,
    history, ["pv_power_kw", "SWGDN", "temperature_c", "cloud_cover"])
enhanced = enhance_windows(w, physics)
clear_kw, clear_ghi = clear_profile(past, physics)
names = list(enhanced.history_feature_names)
ratios = enhanced.gru_history[0, :, names.index("power_clear_ratio")]
ghi_ratios = enhanced.gru_history[0, :, names.index("ghi_clear_ratio")]
np.testing.assert_allclose(ratios, 7./np.maximum(clear_kw, 50.), atol=2e-8, rtol=0)
np.testing.assert_allclose(ghi_ratios, 9./np.maximum(clear_ghi, 20.), atol=2e-8, rtol=0)
assert np.isfinite(enhanced.gru_history).all()
assert (clear_kw < 50).any() and (clear_ghi < 20).any()
np.testing.assert_allclose(ratios[clear_kw < 50], .14, atol=2e-8, rtol=0)
result = {"status": "passed", "hand_computed_horizontal_1000_W_m2_power_kw": 3938.,
    "ambient_temperature_plus_10_C_power_kw": 3762., "capacity_clipped_power_kw": 5000.,
    "night_zero": True, "negative_and_nonfinite_irradiance_rejected": True,
    "low_sun_and_night_nonzero_sensor_history_finite": True,
    "power_ratio_floor_kw": 50., "irradiance_ratio_floor_W_m2": 20.,
    "all_288_history_points_and_96_targets_retained": True}
(out/"physical_boundary_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
