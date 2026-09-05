"""Entry point: land both Small splits in DuckDB and report base rates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from aml.config import ACTIVE_SPLITS, RESULTS
from aml.db import base_rate, connect, load_accounts, load_patterns, load_transactions, timings_frame


def main() -> int:
    pd.set_option("display.width", 140)
    timings = []
    rates = []
    with connect() as con:
        for split in ACTIVE_SPLITS:
            for loader in (load_transactions, load_accounts, load_patterns):
                timing = loader(con, split)
                timings.append(timing)
                print(f"{split:10s} {timing.stage:20s} {timing.rows:>10,} rows  {timing.seconds:7.2f}s")
            rates.append(base_rate(con, split))

    RESULTS.mkdir(parents=True, exist_ok=True)
    timing_table = timings_frame(timings)
    timing_table.to_csv(RESULTS / "timings_load.csv", index=False)

    rate_table = pd.DataFrame(rates)
    rate_table.to_csv(RESULTS / "base_rates.csv", index=False)

    print("\nbase rates")
    print(
        rate_table[
            ["split", "transactions", "laundering", "base_rate_pct", "one_in", "span_days", "distinct_senders"]
        ].to_string(index=False)
    )
    print(f"\nfirst/last timestamps")
    print(rate_table[["split", "first_ts", "last_ts"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
