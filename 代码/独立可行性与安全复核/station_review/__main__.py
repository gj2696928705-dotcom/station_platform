"""JSON CLI: exit 0 pass, 2 rejected, 1 malformed/IO. Standard library only."""
import argparse
import json
from pathlib import Path
import sys

from .adapters import audit_d02, load_json, review_a02_archive
from .examples import fixture
from .review import review_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description="A03 independent offline simulation review")
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review")
    review.add_argument("--request", type=Path, required=True)
    demo = sub.add_parser("demo", help="Explicit synthetic hand example, no external dispatch")
    archive = sub.add_parser("archive-a02")
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--data-root", type=Path, required=True)
    archive.add_argument("--demo-assumptions", action="store_true")
    audit = sub.add_parser("audit-d02")
    audit.add_argument("--data-root", type=Path, required=True)
    for p in (review, demo, archive, audit):
        p.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "review":
            request = load_json(args.request)
            result = review_plan(request["frozen_plan"], request["latest_state"], request["governance_policy"],
                                 as_of=request["as_of"], expected_state_hash=request.get("expected_state_hash"))
            exit_code = 0 if result["passed"] else 2
        elif args.command == "demo":
            b, s, p = fixture()
            result = {"request": {"frozen_plan": b.to_dict(), "latest_state": s, "governance_policy": p, "as_of": s["available_at"]},
                      "review": review_plan(b, s, p, as_of=s["available_at"])}
            exit_code = 0 if result["review"]["passed"] else 2
        elif args.command == "archive-a02":
            result = review_a02_archive(args.archive, args.data_root, demo_assumptions=args.demo_assumptions)
            exit_code = 0 if result["passed_count"] == len(result["scenarios"]) else 2
        else:
            result, exit_code = audit_d02(args.data_root), 2
    except (OSError, KeyError, TypeError, ValueError) as e:
        result = {"passed": False, "device_dispatch_authorized": False,
                  "violations": [{"code": getattr(e, "code", "CLI_INPUT_INVALID"), "detail": str(e)}]}
        exit_code = 1
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
