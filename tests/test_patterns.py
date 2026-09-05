"""Tests for the pattern file parser. These run with no network and no raw data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.config import PATTERN_COLUMNS
from aml.patterns import PatternParseError, parse_pattern_file, summarise_typologies

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_Patterns.txt"


@pytest.fixture(scope="module")
def rings():
    return parse_pattern_file(FIXTURE, split="SAMPLE")


def test_row_count_matches_transaction_lines(rings):
    assert len(rings) == 8


def test_ring_count_and_typologies(rings):
    assert rings["ring_key"].nunique() == 3
    assert set(rings["typology"]) == {"CYCLE", "FAN-OUT", "RANDOM"}


def test_ring_ids_are_contiguous_from_zero(rings):
    assert sorted(rings["ring_id"].unique().tolist()) == [0, 1, 2]


def test_block_header_metadata_is_captured(rings):
    cycle_meta = rings.loc[rings["typology"] == "CYCLE", "ring_meta"].unique().tolist()
    assert cycle_meta == ["Max 10 hops"]


def test_block_without_metadata_parses(rings):
    random_meta = rings.loc[rings["typology"] == "RANDOM", "ring_meta"].unique().tolist()
    assert random_meta == [""]


def test_all_transaction_columns_present(rings):
    for column in PATTERN_COLUMNS:
        assert column in rings.columns


def test_cross_currency_row_preserves_both_currencies(rings):
    row = rings.loc[rings["currency_received"] == "Yuan"].iloc[0]
    assert row["currency_paid"] == "US Dollar"
    assert row["amount_received"] != row["amount_paid"]


def test_numeric_columns_are_numeric(rings):
    assert rings["amount_received"].dtype.kind == "f"
    assert rings["from_bank"].dtype.kind in "iu"
    assert set(rings["is_laundering"].unique()) == {1}


def test_summary_table_totals(rings):
    summary = summarise_typologies(rings)
    total = summary.loc[summary["typology"] == "total"].iloc[0]
    assert total["rings"] == 3
    assert total["transactions"] == 8
    cycle = summary.loc[summary["typology"] == "CYCLE"].iloc[0]
    assert cycle["rings"] == 1
    assert cycle["median_ring_size"] == 3


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "broken_Patterns.txt"
    path.write_text(body)
    return path


def test_unterminated_block_raises(tmp_path):
    body = (
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops\n"
        "2022/09/01 00:20,10,A,10,B,1.0,US Dollar,1.0,US Dollar,Cheque,1\n"
    )
    with pytest.raises(PatternParseError, match="never closed"):
        parse_pattern_file(_write(tmp_path, body))


def test_mismatched_end_typology_raises(tmp_path):
    body = (
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops\n"
        "2022/09/01 00:20,10,A,10,B,1.0,US Dollar,1.0,US Dollar,Cheque,1\n"
        "END LAUNDERING ATTEMPT - STACK\n"
    )
    with pytest.raises(PatternParseError, match="closes"):
        parse_pattern_file(_write(tmp_path, body))


def test_transaction_outside_block_raises(tmp_path):
    body = "2022/09/01 00:20,10,A,10,B,1.0,US Dollar,1.0,US Dollar,Cheque,1\n"
    with pytest.raises(PatternParseError, match="outside any block"):
        parse_pattern_file(_write(tmp_path, body))


def test_wrong_field_count_raises(tmp_path):
    body = (
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops\n"
        "2022/09/01 00:20,10,A,10,B,1.0,US Dollar\n"
        "END LAUNDERING ATTEMPT - CYCLE\n"
    )
    with pytest.raises(PatternParseError, match="expected 11 fields"):
        parse_pattern_file(_write(tmp_path, body))


def test_nested_begin_raises(tmp_path):
    body = (
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops\n"
        "BEGIN LAUNDERING ATTEMPT - STACK:  Max 3 layers\n"
        "END LAUNDERING ATTEMPT - STACK\n"
    )
    with pytest.raises(PatternParseError, match="nested BEGIN"):
        parse_pattern_file(_write(tmp_path, body))
