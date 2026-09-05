"""Descriptive profile of the eight planted typologies.

This is what shows the shapes are genuinely different from each other before
any detector is built. Amounts are never summed across currencies, so the
money columns here are counts of distinct currencies rather than totals.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def ring_profile(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    """One row per ring: size, span, account count, currency count."""
    s = split.replace("-", "_").lower()
    return con.execute(
        f"""
        WITH ring AS (
            SELECT r.ring_key, r.typology, t.*
            FROM ringmap_{s} r JOIN trans_{s} t USING (txn_id)
        )
        SELECT
            ring_key,
            typology,
            count(*) AS ring_size,
            epoch(max(ts) - min(ts)) / 86400.0 AS span_days,
            count(DISTINCT from_account) AS distinct_senders,
            count(DISTINCT to_account) AS distinct_receivers,
            count(DISTINCT payment_format) AS distinct_formats,
            min(ts) AS first_ts,
            max(ts) AS last_ts
        FROM ring
        GROUP BY ring_key, typology
        """
    ).df()


def ring_account_counts(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    """Distinct accounts and distinct currencies touched by each ring."""
    s = split.replace("-", "_").lower()
    return con.execute(
        f"""
        WITH ring AS (
            SELECT r.ring_key, t.from_account, t.to_account, t.currency_received, t.currency_paid
            FROM ringmap_{s} r JOIN trans_{s} t USING (txn_id)
        ),
        accts AS (
            SELECT ring_key, from_account AS account FROM ring
            UNION
            SELECT ring_key, to_account FROM ring
        ),
        curr AS (
            SELECT ring_key, currency_received AS currency FROM ring
            UNION
            SELECT ring_key, currency_paid FROM ring
        )
        SELECT
            a.ring_key,
            count(DISTINCT a.account) AS distinct_accounts,
            (SELECT count(DISTINCT c.currency) FROM curr c WHERE c.ring_key = a.ring_key) AS distinct_currencies
        FROM accts a
        GROUP BY a.ring_key
        """
    ).df()


def account_reuse(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    """How often an account appears in more than one ring, per typology."""
    s = split.replace("-", "_").lower()
    return con.execute(
        f"""
        WITH ring AS (
            SELECT r.ring_key, r.typology, t.from_account, t.to_account
            FROM ringmap_{s} r JOIN trans_{s} t USING (txn_id)
        ),
        accts AS (
            SELECT ring_key, typology, from_account AS account FROM ring
            UNION
            SELECT ring_key, typology, to_account FROM ring
        ),
        per_account AS (
            SELECT account, count(DISTINCT ring_key) AS n_rings FROM accts GROUP BY account
        )
        SELECT
            a.typology,
            count(DISTINCT a.account) AS accounts,
            count(DISTINCT a.account) FILTER (p.n_rings > 1) AS accounts_in_multiple_rings,
            round(100.0 * count(DISTINCT a.account) FILTER (p.n_rings > 1) / count(DISTINCT a.account), 1)
                AS reuse_pct
        FROM accts a JOIN per_account p USING (account)
        GROUP BY a.typology
        ORDER BY a.typology
        """
    ).df()


def typology_table(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    """The committed typology description table for one split."""
    profile = ring_profile(con, split).merge(ring_account_counts(con, split), on="ring_key")
    grouped = (
        profile.groupby("typology")
        .agg(
            rings=("ring_key", "nunique"),
            transactions=("ring_size", "sum"),
            median_ring_size=("ring_size", "median"),
            median_span_days=("span_days", "median"),
            max_span_days=("span_days", "max"),
            median_accounts=("distinct_accounts", "median"),
            median_currencies=("distinct_currencies", "median"),
            max_currencies=("distinct_currencies", "max"),
            median_formats=("distinct_formats", "median"),
        )
        .reset_index()
    )
    reuse = account_reuse(con, split)[["typology", "accounts", "reuse_pct"]]
    table = grouped.merge(reuse, on="typology")
    table.insert(0, "split", split)
    for col in ("median_span_days", "max_span_days"):
        table[col] = table[col].round(2)
    return table
