"""Join pattern file rows back to transaction rows, and verify that join.

The dataset has no transaction identifier, so ring membership has to be matched
on a composite key. Every number in this project sits on this join, so the
verification counts are a committed deliverable rather than a debugging aid.

Two keys are reported. The full key uses every non label field. The minimal key
uses only the fields named in the project brief: timestamp, both banks, both
accounts and both amounts. Reporting both shows how much the currency and
payment format fields contribute to uniqueness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import duckdb
import pandas as pd

FULL_KEY = (
    "ts",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "amount_paid",
    "currency_received",
    "currency_paid",
    "payment_format",
)

MINIMAL_KEY = (
    "ts",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "amount_paid",
)


@dataclass
class JoinReport:
    """Verification counts for one split under one key definition."""

    split: str
    key: str
    pattern_rows: int
    matched_exactly_one: int
    matched_multiple: int
    matched_none: int
    total_matched_txn_rows: int
    seconds: float

    @property
    def match_rate(self) -> float:
        return (self.matched_exactly_one + self.matched_multiple) / self.pattern_rows


def _tables(split: str) -> tuple[str, str]:
    suffix = split.replace("-", "_").lower()
    return f"trans_{suffix}", f"patterns_{suffix}"


def _on_clause(key: tuple[str, ...]) -> str:
    return " AND ".join(f"t.{col} = p.{col}" for col in key)


def build_match_table(
    con: duckdb.DuckDBPyConnection, split: str, key: tuple[str, ...], name: str
) -> JoinReport:
    """Count how many transaction rows each pattern row matches."""
    trans, patterns = _tables(split)
    start = time.perf_counter()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT
            p.pattern_row_id,
            p.ring_key,
            p.typology,
            count(t.txn_id) AS n_matches,
            min(t.txn_id) AS first_txn_id
        FROM {patterns} p
        LEFT JOIN {trans} t ON {_on_clause(key)}
        GROUP BY p.pattern_row_id, p.ring_key, p.typology
        """
    )
    row = con.execute(
        f"""
        SELECT
            count(*) AS pattern_rows,
            count(*) FILTER (n_matches = 1) AS exactly_one,
            count(*) FILTER (n_matches > 1) AS multiple,
            count(*) FILTER (n_matches = 0) AS none_matched,
            coalesce(sum(n_matches), 0) AS total_matched
        FROM {name}
        """
    ).fetchone()
    seconds = time.perf_counter() - start
    return JoinReport(
        split=split,
        key="full" if key is FULL_KEY else "minimal",
        pattern_rows=int(row[0]),
        matched_exactly_one=int(row[1]),
        matched_multiple=int(row[2]),
        matched_none=int(row[3]),
        total_matched_txn_rows=int(row[4]),
        seconds=seconds,
    )


def multiplicity_profile(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """How many pattern rows matched 0, 1, 2, ... transaction rows."""
    return con.execute(
        f"""
        SELECT n_matches, count(*) AS pattern_rows
        FROM {name}
        GROUP BY n_matches
        ORDER BY n_matches
        """
    ).df()


def unmatched_examples(con: duckdb.DuckDBPyConnection, split: str, name: str, limit: int = 10) -> pd.DataFrame:
    """Pattern rows that found no transaction row, for inspection."""
    _, patterns = _tables(split)
    return con.execute(
        f"""
        SELECT p.*
        FROM {name} m
        JOIN {patterns} p USING (pattern_row_id)
        WHERE m.n_matches = 0
        LIMIT {limit}
        """
    ).df()


def unmatched_by_typology(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Match outcome broken down by typology."""
    return con.execute(
        f"""
        SELECT
            typology,
            count(*) AS pattern_rows,
            count(*) FILTER (n_matches = 0) AS unmatched,
            count(*) FILTER (n_matches = 1) AS exactly_one,
            count(*) FILTER (n_matches > 1) AS multiple
        FROM {name}
        GROUP BY typology
        ORDER BY typology
        """
    ).df()


def report_frame(reports: list[JoinReport]) -> pd.DataFrame:
    """Tidy frame of join verification counts for the committed table."""
    return pd.DataFrame(
        [
            {
                "split": r.split,
                "key": r.key,
                "pattern_rows": r.pattern_rows,
                "matched_exactly_one": r.matched_exactly_one,
                "matched_multiple": r.matched_multiple,
                "matched_none": r.matched_none,
                "match_rate_pct": round(100.0 * r.match_rate, 3),
                "total_matched_txn_rows": r.total_matched_txn_rows,
                "seconds": round(r.seconds, 2),
            }
            for r in reports
        ]
    )
