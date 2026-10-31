"""一次完整的 XGBoost 负荷预测训练、评估、预测和可视化验证。

直接运行：
    python 运行.py

调试运行（减少训练轮数和窗口数量）：
    python 运行.py --rounds 30 --stride 24
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from load_forecast.cli import add_training_options, train_from_args
from load_forecast.pipeline import dump, evaluate_bundle, predict_record


def main(argv=None):
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=directory.parent / "数据" / "负荷数据" / "黄花站_2025年_15分钟负荷模拟数据.csv")
    parser.add_argument("--output-dir", type=Path)
    add_training_options(parser)
    args = parser.parse_args(argv)
    args.output_dir = (args.output_dir or directory / "运行结果" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    print(f"负荷预测离线训练（候选模型）\n输入：{args.input.resolve()}\n输出：{args.output_dir}", flush=True)
    try:
        result = train_from_args(args)
        forecast = predict_record(args.output_dir, args.input, args.output_dir / "forecast.json")
        evaluation = evaluate_bundle(args.output_dir, args.input, args.output_dir / "evaluation.json")
        metrics = result["metrics"]["test"]
        errors = (result["metrics"]["visualization_errors"]
                  + forecast["visualization_errors"]
                  + evaluation["visualization_errors"])
        verification = {
            "status": "passed" if forecast["status"] == evaluation["status"] == "available" and not errors else "failed",
            "source": str(args.input.resolve()),
            "source_kind": result["metadata"]["input_quality"]["source_kind"],
            "selected_model": result["metadata"]["selected_model"],
            "forecast_points": len(forecast["forecast_values"]),
            "decision_eligible": False,
            "test_metrics": metrics,
            "visualizations": result["metrics"].get("visualizations", []) + forecast.get("visualizations", []) + evaluation.get("visualizations", []),
            "visualization_errors": errors,
        }
        dump(args.output_dir / "verification.json", verification)
        r2 = "N/A" if metrics.get("r2") is None else f"{metrics['r2']:.4f}"
        print("\n验证完成", flush=True)
        print(f"选中模型：{verification['selected_model']}", flush=True)
        print(f"留出测试集：RMSE={metrics['rmse_kw']:.4f} kW | "
              f"MAE={metrics['mae_kw']:.4f} kW | R²={r2} | "
              f"区间覆盖率={metrics.get('coverage', float('nan')):.2%}", flush=True)
        print(f"未来预测点数：{verification['forecast_points']}", flush=True)
        print(f"日志：{args.output_dir / 'training.log'}", flush=True)
        print(f"指标：{args.output_dir / 'metrics.json'}", flush=True)
        print("图片：", flush=True)
        for path in verification["visualizations"]:
            print(f"  - {args.output_dir / path}", flush=True)
        if errors:
            print("图片或评估错误：", errors, file=sys.stderr, flush=True)
        return 0 if verification["status"] == "passed" else 2
    except Exception as exc:
        # train_model also writes failure.json; this message makes IDE/terminal runs explicit.
        print(f"负荷预测验证失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
