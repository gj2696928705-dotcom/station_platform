"""JSON stdout, training progress stderr. No runtime training in predict."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from .data import InputSpec
from .pipeline import TrainingConfig, evaluate_bundle, predict_record, train_model


def add_training_options(parser):
    parser.add_argument("--target", choices=["base_load_kw", "total_load_kw"], default="base_load_kw")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--source-target-col")
    parser.add_argument("--input-kind", choices=["interval_average_power", "interval_energy"], default="interval_average_power")
    parser.add_argument("--energy-interval-minutes", type=int)
    parser.add_argument("--station-id", default="HH-Huanghua-CSS")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=.05)
    parser.add_argument("--stride", type=int, default=4, help="origin stride in quarter-hours; default 4 = all hourly origins")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")


def train_from_args(args):
    spec = InputSpec(timestamp_col=args.timestamp_col, target_col=args.target, timezone=args.timezone,
                     station_id=args.station_id, source_target_col=args.source_target_col,
                     input_kind=args.input_kind, energy_interval_minutes=args.energy_interval_minutes)
    config = TrainingConfig(rounds=args.rounds, patience=args.patience, max_depth=args.max_depth,
                            learning_rate=args.learning_rate, stride=args.stride, threads=args.threads, seed=args.seed)
    return train_model(args.input, args.output_dir, spec, config,
                       progress=None if args.quiet else lambda line: print(line, file=sys.stderr, flush=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Station load XGBoost candidate model")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    add_training_options(train)
    for command in ("predict", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--bundle-dir", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "predict":
            child.add_argument("--forecast-start")
            child.add_argument("--as-of", help="Explicit replay clock; rejects stale or future history")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = train_from_args(args)
        elif args.command == "predict":
            result = predict_record(args.bundle_dir, args.input, args.output, forecast_start=args.forecast_start, as_of=args.as_of)
        else:
            result = evaluate_bundle(args.bundle_dir, args.input, args.output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 2 if result.get("status") == "unavailable" else 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
