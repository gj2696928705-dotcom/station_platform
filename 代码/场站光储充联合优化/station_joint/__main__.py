"""Run from 代码/场站光储充联合优化: python -m station_joint --help."""
import argparse
import json
from pathlib import Path

from .contract import InputError
from .replay import LinkedReplay, compare_idle


def main():
    parser = argparse.ArgumentParser(description="A02 four-slot offline joint optimization; never dispatches devices")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="complete frozen input JSON")
    source.add_argument("--data-root", type=Path, help="代码/数据 directory; replays all D01 scenarios")
    parser.add_argument("--demo-assumptions", action="store_true", help="explicit hypothetical existing profiles, costs and window terminal rule")
    parser.add_argument("--time-limit", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data_root:
        root = args.data_root.resolve()
        output = args.output.resolve()
        if output.is_relative_to(root):
            parser.error("write evidence outside original data packages")
        adapter = LinkedReplay(root)
        rows = []
        for sid in adapter.scenario_ids():
            try:
                d = adapter.load(sid, demo_assumptions=args.demo_assumptions)
                rows.append({"scenario_id": sid, "input": d, **compare_idle(d, time_limit_seconds=args.time_limit,
                                                                          allow_assumptions=args.demo_assumptions)})
            except InputError as e:
                rows.append({"scenario_id": sid, "status": "INPUT_UNAVAILABLE", "reason": e.code, "detail": str(e)})
        result = {"simulation": True, "execution_mode": "offline_replay", "demo_assumptions": args.demo_assumptions,
                  "scenarios": rows, "source_refs": adapter.refs}
    else:
        if args.input.resolve() == args.output.resolve():
            parser.error("output must not overwrite frozen input")
        d = json.loads(args.input.read_text(encoding="utf-8-sig"))
        result = {"input": d, **compare_idle(d, time_limit_seconds=args.time_limit, allow_assumptions=args.demo_assumptions)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if "scenarios" in result:
        print(json.dumps([{ "scenario_id": r["scenario_id"], "status": r.get("status", r.get("joint", {}).get("status")),
                           "reason": r.get("reason"), "delta": r.get("delta_joint_minus_idle")} for r in result["scenarios"]], ensure_ascii=False))
    else:
        print(result["joint"]["status"])


if __name__ == "__main__":
    main()
