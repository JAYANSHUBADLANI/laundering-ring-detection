"""Tests for the dense period boundary and the temporal cut.

These build a tiny in memory table rather than touching the real database, so
they run with no raw data and no network.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml import periods


def build(rows):
    """One in memory transactions table shaped like a real split."""
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE trans_test_split (txn_id BIGINT, ts TIMESTAMP, is_laundering TINYINT)"
    )
    con.executemany(
        "INSERT INTO trans_test_split VALUES (?, ?, ?)",
        [(i, ts, lab) for i, (ts, lab) in enumerate(rows)],
    )
    return con


def dense_then_collapse(dense_days=5, dense_per_day=1000, tail_days=3, tail_per_day=2):
    """A file that looks like the real ones: steady traffic, then a collapse."""
    start = datetime(2022, 9, 1)
    rows = []
    for day in range(dense_days):
        for i in range(dense_per_day):
            rows.append((start + timedelta(days=day, seconds=i), 0))
    for day in range(dense_days, dense_days + tail_days):
        for i in range(tail_per_day):
            rows.append((start + timedelta(days=day, seconds=i), 1))
    return rows


def test_dense_end_lands_on_the_first_collapsed_day():
    con = build(dense_then_collapse())
    result = periods.compute(con, "test-split")
    assert result.dense_end == datetime(2022, 9, 6)


def test_tail_is_excluded_from_train_and_test():
    con = build(dense_then_collapse())
    p = periods.compute(con, "test-split")
    total = con.execute("SELECT count(*) FROM trans_test_split").fetchone()[0]
    counted = 0
    for clause in (p.train_clause, p.test_clause):
        counted += con.execute(
            f"SELECT count(*) FROM trans_test_split WHERE {clause}"
        ).fetchone()[0]
    tail = con.execute(
        f"SELECT count(*) FROM trans_test_split WHERE {p.tail_clause}"
    ).fetchone()[0]
    assert counted + tail == total
    assert tail == 6


def test_train_and_test_do_not_overlap():
    con = build(dense_then_collapse())
    p = periods.compute(con, "test-split")
    overlap = con.execute(
        f"SELECT count(*) FROM trans_test_split "
        f"WHERE ({p.train_clause}) AND ({p.test_clause})"
    ).fetchone()[0]
    assert overlap == 0


def test_cut_sits_near_the_configured_quantile():
    con = build(dense_then_collapse())
    p = periods.compute(con, "test-split")
    before = con.execute(
        f"SELECT count(*) FROM trans_test_split WHERE {p.train_clause}"
    ).fetchone()[0]
    dense_total = con.execute(
        f"SELECT count(*) FROM trans_test_split WHERE ts < TIMESTAMP '{p.dense_end}'"
    ).fetchone()[0]
    assert before / dense_total == pytest.approx(0.70, abs=0.01)


def test_a_file_with_no_collapse_has_no_tail():
    """A split that never breaks must not have a boundary invented for it."""
    con = build(dense_then_collapse(tail_days=0))
    p = periods.compute(con, "test-split")
    tail = con.execute(
        f"SELECT count(*) FROM trans_test_split WHERE {p.tail_clause}"
    ).fetchone()[0]
    assert tail == 0
