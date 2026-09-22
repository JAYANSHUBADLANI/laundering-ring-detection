"""Tests for the composite key join between pattern rows and transactions.

Every number in the project rests on this join. On the real files it is checked
by the counts `scripts/verify_join.py` reports; these build a handful of rows in
memory where the right answer is known by reading, so they run with no raw data
and no network.
"""

from __future__ import annotations

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml import join

T0 = datetime(2022, 9, 1, 10, 0)
T1 = datetime(2022, 9, 1, 10, 5)
T2 = datetime(2022, 9, 1, 11, 0)

TRANS_SCHEMA = """
    txn_id BIGINT, ts TIMESTAMP,
    from_bank BIGINT, from_account VARCHAR, to_bank BIGINT, to_account VARCHAR,
    amount_received DECIMAL(20,2), currency_received VARCHAR,
    amount_paid DECIMAL(20,2), currency_paid VARCHAR,
    payment_format VARCHAR, is_laundering TINYINT
"""

PATTERN_SCHEMA = """
    pattern_row_id BIGINT, ring_key VARCHAR, typology VARCHAR, ts TIMESTAMP,
    from_bank BIGINT, from_account VARCHAR, to_bank BIGINT, to_account VARCHAR,
    amount_received DECIMAL(20,2), currency_received VARCHAR,
    amount_paid DECIMAL(20,2), currency_paid VARCHAR,
    payment_format VARCHAR, is_laundering TINYINT
"""


def leg(ts, src, dst, amount, fmt="Wire", currency="US Dollar"):
    """One transfer, in the column order shared by both tables after the ids."""
    amount = Decimal(amount)
    return (ts, 1, src, 2, dst, amount, currency, amount, currency, fmt, 1)


def build(trans_legs, pattern_legs):
    """In memory tables named the way the join module expects for one split."""
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE trans_test_split ({TRANS_SCHEMA})")
    con.execute(f"CREATE TABLE patterns_test_split ({PATTERN_SCHEMA})")
    con.executemany(
        "INSERT INTO trans_test_split VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(i + 1, *row) for i, row in enumerate(trans_legs)],
    )
    con.executemany(
        "INSERT INTO patterns_test_split VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(i, ring, typ, *row) for i, (ring, typ, row) in enumerate(pattern_legs)],
    )
    return con


def cycle_ring():
    """A three leg cycle A to B to C to A, present once in the transactions."""
    return [
        leg(T0, "A", "B", "100.00"),
        leg(T1, "B", "C", "99.00"),
        leg(T2, "C", "A", "98.00"),
    ]


def test_every_ring_leg_matches_exactly_one_transaction():
    legs = cycle_ring()
    background = [leg(T0, "X", "Y", "5.00"), leg(T1, "Y", "Z", "7.50")]
    con = build(legs + background, [("r1", "CYCLE", row) for row in legs])
    report = join.build_match_table(con, "test-split", join.FULL_KEY, "m")
    assert report.pattern_rows == 3
    assert report.matched_exactly_one == 3
    assert report.matched_multiple == 0
    assert report.matched_none == 0
    assert report.total_matched_txn_rows == 3
    assert report.match_rate == 1.0
    assert report.key == "full"


def test_a_leg_missing_from_the_transactions_is_counted_as_unmatched():
    legs = cycle_ring()
    con = build(legs[:2], [("r1", "CYCLE", row) for row in legs])
    report = join.build_match_table(con, "test-split", join.FULL_KEY, "m")
    assert report.matched_exactly_one == 2
    assert report.matched_none == 1
    assert report.match_rate == pytest.approx(2 / 3)
    missing = join.unmatched_examples(con, "test-split", "m")
    assert list(missing["from_account"]) == ["C"]


def test_one_cent_difference_does_not_match():
    """Amounts are exact decimals, so a near miss must not be absorbed."""
    pattern = leg(T0, "A", "B", "100.00")
    con = build([leg(T0, "A", "B", "100.01")], [("r1", "FAN-OUT", pattern)])
    report = join.build_match_table(con, "test-split", join.FULL_KEY, "m")
    assert report.matched_none == 1


def test_duplicate_transactions_are_reported_as_multiple_matches():
    pattern = leg(T0, "A", "B", "100.00")
    con = build([pattern, pattern], [("r1", "FAN-OUT", pattern)])
    report = join.build_match_table(con, "test-split", join.FULL_KEY, "m")
    assert report.matched_exactly_one == 0
    assert report.matched_multiple == 1
    assert report.total_matched_txn_rows == 2
    # Multiple matches still count toward the match rate, which is why the
    # committed table reports them separately.
    assert report.match_rate == 1.0
    profile = join.multiplicity_profile(con, "m")
    assert profile.to_dict("records") == [{"n_matches": 2, "pattern_rows": 1}]


def test_payment_format_separates_rows_the_minimal_key_cannot():
    wire = leg(T0, "A", "B", "100.00", fmt="Wire")
    cash = leg(T0, "A", "B", "100.00", fmt="Cash")
    con = build([wire, cash], [("r1", "FAN-OUT", wire)])
    full = join.build_match_table(con, "test-split", join.FULL_KEY, "m_full")
    minimal = join.build_match_table(con, "test-split", join.MINIMAL_KEY, "m_min")
    assert (full.matched_exactly_one, full.matched_multiple) == (1, 0)
    assert (minimal.matched_exactly_one, minimal.matched_multiple) == (0, 1)
    assert minimal.key == "minimal"


def test_both_amounts_are_part_of_the_minimal_key():
    """A cross currency leg is only pinned down by the amount paid as well."""
    ring_leg = (T0, 1, "A", 2, "B", Decimal("100.00"), "Euro", Decimal("108.00"), "US Dollar", "Wire", 1)
    other = (T0, 1, "A", 2, "B", Decimal("100.00"), "Euro", Decimal("100.00"), "Euro", "Wire", 0)
    con = build([ring_leg, other], [("r1", "FAN-OUT", ring_leg)])
    report = join.build_match_table(con, "test-split", join.MINIMAL_KEY, "m")
    assert (report.matched_exactly_one, report.matched_multiple) == (1, 0)


COLUMNS = (
    "ts", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "currency_received", "amount_paid", "currency_paid",
    "payment_format", "is_laundering",
)
CHANGED = {
    "ts": T1,
    "from_bank": 9,
    "from_account": "Q",
    "to_bank": 9,
    "to_account": "Q",
    "amount_received": Decimal("1.00"),
    "currency_received": "Yen",
    "amount_paid": Decimal("1.00"),
    "currency_paid": "Yen",
    "payment_format": "Cash",
}


# Written out by hand rather than read from the module, so dropping a column
# from either key fails a test instead of quietly shrinking the test list.
EXPECTED_MINIMAL = (
    "ts", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "amount_paid",
)
EXPECTED_FULL = EXPECTED_MINIMAL + ("currency_received", "currency_paid", "payment_format")


@pytest.mark.parametrize(
    "key, column",
    [(join.FULL_KEY, c) for c in EXPECTED_FULL] + [(join.MINIMAL_KEY, c) for c in EXPECTED_MINIMAL],
)
def test_changing_any_single_key_column_breaks_the_match(key, column):
    ring_leg = leg(T0, "A", "B", "100.00")
    decoy = list(ring_leg)
    decoy[COLUMNS.index(column)] = CHANGED[column]
    con = build([tuple(decoy)], [("r1", "FAN-OUT", ring_leg)])
    report = join.build_match_table(con, "test-split", key, "m")
    assert report.matched_none == 1


def test_breakdown_by_typology_adds_up():
    ring = cycle_ring()
    hub = leg(T0, "H", "S1", "40.00")
    con = build(
        ring[:2] + [hub],
        [("r1", "CYCLE", row) for row in ring] + [("r2", "FAN-OUT", hub)],
    )
    join.build_match_table(con, "test-split", join.FULL_KEY, "m")
    table = join.unmatched_by_typology(con, "m").set_index("typology")
    assert table.loc["CYCLE", "pattern_rows"] == 3
    assert table.loc["CYCLE", "unmatched"] == 1
    assert table.loc["FAN-OUT", "exactly_one"] == 1
    assert int(table["pattern_rows"].sum()) == 4


def test_report_frame_rounds_the_match_rate_to_a_percentage():
    legs = cycle_ring()
    con = build(legs[:2], [("r1", "CYCLE", row) for row in legs])
    report = join.build_match_table(con, "test-split", join.FULL_KEY, "m")
    frame = join.report_frame([report])
    assert frame.loc[0, "match_rate_pct"] == 66.667
    assert frame.loc[0, "split"] == "test-split"
