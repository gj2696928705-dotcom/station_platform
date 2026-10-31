"""前三天39点功率 + 当天辐照/降雨，预测07:00–19:00的13个小时点。"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from pv_forecast.data import InputSpec
from pv_forecast.day13 import Day13TrainingConfig
from pv_forecast.day13_data import prepare_hourly, read_day_frame
from pv_forecast.pipeline import train_model, predict_record, evaluate_bundle


def main(argv=None):
    folder = Path(__file__).resolve().parent
    data_folder = folder.parent/"数据"/"光伏数据"
    default_input = data_folder/"黄花站_2025年_07-19_小时数据.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--future-weather-input", type=Path, help="可选，包含issued_at/timestamp及四列气象的小时预报归档")
    parser.add_argument("--forecast-date", help="YYYY-MM-DD；省略时仅展示数据末日的回测示例，实际当天预测使用CLI predict")
    args = parser.parse_args(argv)
    spec = InputSpec()
    if args.input == default_input and not default_input.exists():
        hourly = prepare_hourly(data_folder/"黄花站_2025年_15分钟光伏模拟数据.csv")
        hourly.to_csv(default_input, index=False, encoding="utf-8-sig")
    output = args.output_dir or folder/"运行结果"/("day13_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    config = Day13TrainingConfig(gru_epochs=args.epochs, gru_batch_size=args.batch_size,
        gru_patience=args.patience, gru_units=args.units, gru_learning_rate=args.learning_rate)
    print("历史输入39×1（前三天功率），未来输入13×4（SWGDN/DHI/DNI/rain），输出13点、60分钟。", flush=True)
    train_model(args.input, output, spec, config, progress=lambda message: print(message, flush=True),
        future_weather_source=args.future_weather_input)
    date = args.forecast_date or read_day_frame(args.input).index[-1].date().isoformat()
    print(f"生成 {date} 的07:00–19:00预测；默认数据末日属于回测示例。", flush=True)
    forecast = predict_record(output, args.input, output/"forecast.json", forecast_start=date,
        future_weather_source=args.future_weather_input)
    evaluation = evaluate_bundle(output, args.input, visualizations_dir=output/"visualizations",
        future_weather_source=args.future_weather_input)
    (output/"evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"完成，预测状态：{forecast['status']}；产物：{output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
