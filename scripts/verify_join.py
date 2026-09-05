"""Entry point: run and report the pattern to transaction join verification."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from aml.config import ACTIVE_SPLITS, RESULTS
from aml.db import connect
from aml.join import (
    FULL_KEY,
    MINIMAL_KEY,
    build_match_table,
    multiplicity_profile,
    report_frame,
    unmatched_by_typology,
    unmatched_examples,
)


def main() -> int:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    reports = []
    with connect() as con:
        for split in ACTIVE_SPLITS:
            suffix = split.replace("-", "_").lower()
            for key, label in ((FULL_KEY, "full"), (MINIMAL_KEY, "minimal")):
                name = f"match_{label}_{suffix}"
                report = build_match_table(con, split, key, name)
                reports.append(report)
                print(
                    f"{split:10s} {label:8s} rows={report.pattern_rows:>5,} "
                    f"one={report.matched_exactly_one:>5,} "
                    f"multi={report.matched_multiple:>5,} "
                    f"none={report.matched_none:>5,} "
                    f"rate={100 * report.match_rate:6.2f}%  {report.seconds:5.2f}s"
                )

            full_name = f"match_full_{suffix}"
            print(f"\n{split} multiplicity profile, full key")
            print(multiplicity_profile(con, full_name).to_string(index=False))
            print(f"\n{split} by typology, full key")
            print(unmatched_by_typology(con, full_name).to_string(index=False))
            unmatched = unmatched_examples(con, split, full_name, limit=5)
            if len(unmatched):
                print(f"\n{split} unmatched examples")
                print(unmatched.to_string(index=False))
            print()

    RESULTS.mkdir(parents=True, exist_ok=True)
    table = report_frame(reports)
    table.to_csv(RESULTS / "join_verification.csv", index=False)
    print("join verification table")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
