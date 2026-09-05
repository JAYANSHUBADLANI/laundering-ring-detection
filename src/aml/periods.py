"""Where the dense period ends and where the temporal cut falls.

Both decisions are derived from the data by a rule stated in config rather than
written as dates, so that pointing the project at a Medium or Large split does
not require a second judgement call. Both are computed before any model is
fitted and neither depends on a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb

from .config import DENSE_DAY_MIN_SHARE, TEMPORAL_CUT_QUANTILE


@dataclass(frozen=True)
class Periods:
    """The three timestamps that define every evaluation window for one split."""

    split: str
    dense_end: datetime      # exclusive: the dense period is ts < dense_end
    cut: datetime            # train is ts < cut, test is cut <= ts < dense_end
    last_ts: datetime        # end of the whole file, tail included

    @property
    def train_clause(self) -> str:
        return f"ts < TIMESTAMP '{self.cut}'"

    @property
    def test_clause(self) -> str:
        return f"ts >= TIMESTAMP '{self.cut}' AND ts < TIMESTAMP '{self.dense_end}'"

    @property
    def tail_clause(self) -> str:
        return f"ts >= TIMESTAMP '{self.dense_end}'"


def table_for(split: str) -> str:
    return f"trans_{split.replace('-', '_').lower()}"


def compute(con: duckdb.DuckDBPyConnection, split: str) -> Periods:
    """Dense period boundary and temporal cut for one split.

    The boundary is the first day that falls below DENSE_DAY_MIN_SHARE of the
    median daily volume, and everything from that day onward is the tail. The
    cut is the TEMPORAL_CUT_QUANTILE quantile of transaction time inside the
    dense period, so train and test are both drawn from ordinary traffic.
    """
    table = table_for(split)
    daily = con.execute(
        f"""
        SELECT date_trunc('day', ts) AS day, count(*) AS n
        FROM {table} GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    counts = sorted(row[1] for row in daily)
    median = counts[len(counts) // 2]
    threshold = median * DENSE_DAY_MIN_SHARE

    dense_end = None
    for day, n in daily:
        if n < threshold:
            dense_end = day
            break
    if dense_end is None:
        # No break in this split: the dense period is the whole file.
        dense_end = con.execute(f"SELECT max(ts) + INTERVAL 1 SECOND FROM {table}").fetchone()[0]

    # The quantile is taken on the timestamp column directly. An earlier version
    # went through epoch seconds and back via to_timestamp, which returns a value
    # in the session's local zone: dropping the tzinfo from that kept the shifted
    # wall clock and moved the cut by the machine's UTC offset, five and a half
    # hours on the machine this was written on. The unit test on the cut position
    # is what caught it.
    cut_ts = con.execute(
        f"""
        SELECT quantile_cont(ts, {TEMPORAL_CUT_QUANTILE})
        FROM {table} WHERE ts < TIMESTAMP '{dense_end}'
        """
    ).fetchone()[0]

    last_ts = con.execute(f"SELECT max(ts) FROM {table}").fetchone()[0]
    return Periods(split=split, dense_end=dense_end, cut=cut_ts, last_ts=last_ts)
