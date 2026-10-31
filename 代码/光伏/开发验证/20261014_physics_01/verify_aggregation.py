"""Verify pooling on previously seen truth; never load the new final paths."""
from pathlib import Path
import json

out = Path(__file__).resolve().parent
pv = out.parents[1]
script = out/"aggregate_frozen.py"
prefix = script.read_text(encoding="utf-8").split('\nfolder = out/"frozen_three_paths"')[0]
namespace = {"__file__": str(script)}
exec(compile(prefix, str(script), "exec"), namespace)
np, pd = namespace["np"], namespace["pd"]
read, moments, pool = namespace["read"], namespace["moments"], namespace["pool"]
report = read(out/"control_error_diagnosis.json")["models"]["20261009_point_correction_01"]
source = pv/"开发验证"/"20261009_point_correction"/"frozen_2026_seed20261010"/"黄花站_2026年_15分钟_点预测修正冻结模拟数据.csv"
station = read(pv.parent/"数据"/"光伏数据"/"simulation_config.json")["station"]
truth = moments(pd.read_csv(source), station)
pairs = [(report["overall"][key], truth[key]) for key in ("all_day", "daylight", "night")]
pairs += [(values["daylight"], truth["month_"+month]) for month, values in report["months"].items()]
pairs += [(values, truth["weather_"+state]) for state, values in report["daylight_weather"].items()]
pairs += [(values, truth["lead_"+lead]) for lead, values in report["error_diagnostics"]["daylight_by_lead"].items()]
for score, exact in pairs:
    for repeats in (1, 3):
        combined = pool([score]*repeats, [exact]*repeats)
        assert combined["count"] == score["count"]*repeats
        for key in ("rmse", "mae", "r2"):
            if score[key] is not None:
                np.testing.assert_allclose(score[key], combined[key], atol=3e-8, rtol=0)
for subsets in (
    [(values, truth["weather_"+state]) for state, values in report["daylight_weather"].items()],
    [(values["daylight"], truth["month_"+month]) for month, values in report["months"].items()]):
    mixed = pool([score for score, _ in subsets], [exact for _, exact in subsets])
    assert mixed["count"] == report["overall"]["daylight"]["count"]
    for key in ("rmse", "mae", "r2"):
        np.testing.assert_allclose(mixed[key], report["overall"]["daylight"][key], atol=3e-8, rtol=0)
assert not (out/"frozen_three_paths").exists()
result = {"status": "passed", "groups": len(pairs), "repetitions_tested": [1, 3],
    "source": "already seen 2026 seed20261010", "new_final_paths_loaded": False,
    "counts_and_exact_repeated_float32_target_moments_match": True,
    "heterogeneous_month_and_weather_groups_reconstruct_overall_scores": True,
    "pooled_SSE_SAE_R2_match_independent_existing_report": True}
(out/"aggregation_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
