"""Entry point: make sure the raw files for the active splits are on disk."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.config import ACTIVE_SPLITS
from aml.fetch import RawDataMissing, ensure, kaggle_available, status


def main() -> int:
    print(f"kaggle cli available: {kaggle_available()}")
    for row in status(ACTIVE_SPLITS):
        mark = "present" if row.present else "MISSING"
        size = f"{row.size_mb:8.1f} MB" if row.present else " " * 11
        print(f"  {mark:8s} {size}  {row.filename}")

    try:
        ensure(ACTIVE_SPLITS)
    except RawDataMissing as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print("\nall raw files for the active splits are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
