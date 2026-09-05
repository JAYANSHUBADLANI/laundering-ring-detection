"""DuckDB loading for the transaction, account and pattern tables.

Everything at row scale stays in DuckDB. Only small result sets are pulled into
pandas. The transaction CSV header repeats the name Account for both the
sending and receiving side, so the header row is skipped and column names are
supplied explicitly.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from .config import DATA_INTERIM, DATA_RAW, DUCKDB_PATH, SPLITS
from .patterns import parse_pattern_file

TRANS_COLUMNS = {
    "ts_raw": "VARCHAR",
    "from_bank_raw": "VARCHAR",
    "from_account": "VARCHAR",
    "to_bank_raw": "VARCHAR",
    "to_account": "VARCHAR",
    "amount_received": "DECIMAL(20,2)",
    "currency_received": "VARCHAR",
    "amount_paid": "DECIMAL(20,2)",
    "currency_paid": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "TINYINT",
}

ACCOUNT_COLUMNS = {
    "bank_name": "VARCHAR",
    "bank_id_raw": "VARCHAR",
    "account": "VARCHAR",
    "entity_id": "VARCHAR",
    "entity_name": "VARCHAR",
}

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


@dataclass
class StageTiming:
    """Wall clock for one named stage on one split."""

    split: str
    stage: str
    seconds: float
    rows: int


@contextmanager
def connect(path: Path = DUCKDB_PATH, read_only: bool = False):
    """Open the project database, creating the interim directory if needed."""
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def _column_spec(columns: dict[str, str]) -> str:
    inner = ", ".join(f"'{name}': '{dtype}'" for name, dtype in columns.items())
    return "{" + inner + "}"


def load_transactions(con: duckdb.DuckDBPyConnection, split: str) -> StageTiming:
    """Load one split's transaction CSV into a DuckDB table.

    Bank identifiers carry leading zeros in the raw file and must be normalised
    the same way on both sides of the pattern join, so they are cast to BIGINT
    here and in the pattern table.
    """
    path = DATA_RAW / SPLITS[split].trans_file
    table = f"trans_{split.replace('-', '_').lower()}"
    start = time.perf_counter()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            row_number() OVER () AS txn_id,
            strptime(ts_raw, '{TIMESTAMP_FORMAT}') AS ts,
            CAST(from_bank_raw AS BIGINT) AS from_bank,
            from_account,
            CAST(to_bank_raw AS BIGINT) AS to_bank,
            to_account,
            amount_received,
            currency_received,
            amount_paid,
            currency_paid,
            payment_format,
            is_laundering
        FROM read_csv(
            '{path.as_posix()}',
            header = false,
            skip = 1,
            columns = {_column_spec(TRANS_COLUMNS)}
        )
        """
    )
    rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return StageTiming(split, "load_transactions", time.perf_counter() - start, rows)


def load_accounts(con: duckdb.DuckDBPyConnection, split: str) -> StageTiming:
    """Load one split's account file."""
    path = DATA_RAW / SPLITS[split].accounts_file
    table = f"accounts_{split.replace('-', '_').lower()}"
    start = time.perf_counter()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            bank_name,
            CAST(bank_id_raw AS BIGINT) AS bank_id,
            account,
            entity_id,
            entity_name
        FROM read_csv(
            '{path.as_posix()}',
            header = false,
            skip = 1,
            columns = {_column_spec(ACCOUNT_COLUMNS)}
        )
        """
    )
    rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return StageTiming(split, "load_accounts", time.perf_counter() - start, rows)


def load_patterns(con: duckdb.DuckDBPyConnection, split: str) -> StageTiming:
    """Parse the pattern file and land it in DuckDB with matching column types."""
    path = DATA_RAW / SPLITS[split].patterns_file
    table = f"patterns_{split.replace('-', '_').lower()}"
    start = time.perf_counter()
    rings = parse_pattern_file(path, split=split)
    rings = rings.reset_index(drop=True)
    rings["pattern_row_id"] = rings.index
    con.register("rings_frame", rings)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            pattern_row_id,
            split,
            ring_key,
            ring_id,
            typology,
            ring_meta,
            source_line,
            strptime(timestamp, '{TIMESTAMP_FORMAT}') AS ts,
            CAST(from_bank AS BIGINT) AS from_bank,
            from_account,
            CAST(to_bank AS BIGINT) AS to_bank,
            to_account,
            CAST(amount_received AS DECIMAL(20,2)) AS amount_received,
            currency_received,
            CAST(amount_paid AS DECIMAL(20,2)) AS amount_paid,
            currency_paid,
            payment_format,
            CAST(is_laundering AS TINYINT) AS is_laundering
        FROM rings_frame
        """
    )
    con.unregister("rings_frame")
    rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return StageTiming(split, "load_patterns", time.perf_counter() - start, rows)


def base_rate(con: duckdb.DuckDBPyConnection, split: str) -> dict:
    """Transaction count, laundering count and base rate for one split."""
    table = f"trans_{split.replace('-', '_').lower()}"
    row = con.execute(
        f"""
        SELECT
            count(*) AS transactions,
            sum(is_laundering) AS laundering,
            min(ts) AS first_ts,
            max(ts) AS last_ts,
            count(DISTINCT from_account) AS distinct_senders,
            count(DISTINCT to_account) AS distinct_receivers
        FROM {table}
        """
    ).fetchone()
    transactions, laundering, first_ts, last_ts, senders, receivers = row
    return {
        "split": split,
        "transactions": int(transactions),
        "laundering": int(laundering),
        "base_rate": laundering / transactions,
        "base_rate_pct": 100.0 * laundering / transactions,
        "one_in": transactions / laundering if laundering else float("nan"),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "span_days": (last_ts - first_ts).total_seconds() / 86400.0,
        "distinct_senders": int(senders),
        "distinct_receivers": int(receivers),
    }


def timings_frame(timings: list[StageTiming]) -> pd.DataFrame:
    """Tidy frame of stage timings for the committed timing table."""
    return pd.DataFrame(
        [{"split": t.split, "stage": t.stage, "seconds": round(t.seconds, 2), "rows": t.rows} for t in timings]
    )
