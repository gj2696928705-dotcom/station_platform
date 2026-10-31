"""Executed only by an independent numerical Python; predict never trains."""
import json
from pathlib import Path
import sys


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if job["kind"] == "load":
        from load_forecast.pipeline import predict_record
        record = predict_record(job["bundle"], job["history"], job["output"], as_of=job["as_of"])
    elif job["kind"] == "pv":
        from pv_forecast.pipeline import predict_record
        record = predict_record(job["bundle"], job["history"], job["output"],
                                forecast_start=job["forecast_date"], future_weather_source=job.get("weather"),
                                asset_scope=job["station_id"])
    else:
        raise ValueError("Unknown candidate kind")
    print(json.dumps({"status": record["status"], "forecast_id": record["forecast_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
