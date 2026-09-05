"""Parser for the <SPLIT>_Patterns.txt laundering pattern files.

The pattern file is not a label column. It is a sequence of delimited blocks,
each block one complete planted laundering structure:

    BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops
    <the transactions making up that cycle, one per line>
    END LAUNDERING ATTEMPT - CYCLE

The parser raises rather than skipping on anything it does not understand,
because every downstream number in this project sits on these blocks.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

from .config import PATTERN_COLUMNS

BEGIN_RE = re.compile(
    r"^BEGIN\s+LAUNDERING\s+ATTEMPT\s*-\s*(?P<typology>[A-Z][A-Z\-]*)\s*(?::\s*(?P<meta>.*))?$"
)
END_RE = re.compile(r"^END\s+LAUNDERING\s+ATTEMPT\s*-\s*(?P<typology>[A-Z][A-Z\-]*)\s*$")


class PatternParseError(ValueError):
    """Raised when the pattern file does not match the documented structure."""


def _read_lines(path: Path) -> list[str]:
    """Read the file, preferring utf-8 and falling back to latin-1."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    raise PatternParseError(f"could not decode {Path(path).name}")


def parse_pattern_file(path: Path, split: str | None = None) -> pd.DataFrame:
    """Parse one pattern file into a tidy frame of ring member transactions.

    Returns one row per transaction line inside a laundering block, carrying
    the ring identifier, the typology, the block header text and the source
    line number alongside the eleven transaction fields.
    """
    lines = _read_lines(path)
    records: list[dict] = []
    ring_id = -1
    open_typology: str | None = None
    open_meta: str = ""
    open_line: int = -1
    n_expected = len(PATTERN_COLUMNS)

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        begin = BEGIN_RE.match(stripped)
        if begin:
            if open_typology is not None:
                raise PatternParseError(
                    f"{Path(path).name}:{line_no} nested BEGIN, block opened at line {open_line} "
                    f"for {open_typology} was never closed"
                )
            open_typology = begin.group("typology")
            open_meta = (begin.group("meta") or "").strip()
            open_line = line_no
            ring_id += 1
            continue

        end = END_RE.match(stripped)
        if end:
            if open_typology is None:
                raise PatternParseError(f"{Path(path).name}:{line_no} END without a matching BEGIN")
            if end.group("typology") != open_typology:
                raise PatternParseError(
                    f"{Path(path).name}:{line_no} END {end.group('typology')} closes "
                    f"BEGIN {open_typology} from line {open_line}"
                )
            open_typology = None
            open_meta = ""
            open_line = -1
            continue

        if open_typology is None:
            raise PatternParseError(
                f"{Path(path).name}:{line_no} transaction line outside any block: {stripped[:120]!r}"
            )

        fields = next(csv.reader(io.StringIO(stripped)))
        if len(fields) != n_expected:
            raise PatternParseError(
                f"{Path(path).name}:{line_no} expected {n_expected} fields, got {len(fields)}: "
                f"{stripped[:120]!r}"
            )

        record = {name: value.strip() for name, value in zip(PATTERN_COLUMNS, fields)}
        record["ring_id"] = ring_id
        record["typology"] = open_typology
        record["ring_meta"] = open_meta
        record["source_line"] = line_no
        records.append(record)

    if open_typology is not None:
        raise PatternParseError(
            f"{Path(path).name}: block {open_typology} opened at line {open_line} never closed"
        )

    if not records:
        raise PatternParseError(f"{Path(path).name}: no transactions parsed")

    frame = pd.DataFrame.from_records(records)
    if split is not None:
        frame.insert(0, "split", split)
        frame["ring_key"] = split + "::" + frame["ring_id"].astype(str)
    else:
        frame["ring_key"] = frame["ring_id"].astype(str)

    for col in ("from_bank", "to_bank", "is_laundering"):
        frame[col] = pd.to_numeric(frame[col], errors="raise", downcast="integer")
    for col in ("amount_received", "amount_paid"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")

    ordered = (
        (["split"] if split is not None else [])
        + ["ring_key", "ring_id", "typology", "ring_meta", "source_line"]
        + list(PATTERN_COLUMNS)
    )
    return frame[ordered]


def summarise_typologies(rings: pd.DataFrame) -> pd.DataFrame:
    """Rings, transactions and median ring size per typology, plus a total row."""
    per_ring = rings.groupby(["ring_key", "typology"], observed=True).size().rename("ring_size")
    per_ring = per_ring.reset_index()

    summary = (
        per_ring.groupby("typology", observed=True)
        .agg(rings=("ring_key", "nunique"), median_ring_size=("ring_size", "median"))
        .join(rings.groupby("typology", observed=True).size().rename("transactions"))
        .reset_index()[["typology", "rings", "transactions", "median_ring_size"]]
        .sort_values("typology")
        .reset_index(drop=True)
    )

    total = pd.DataFrame(
        [
            {
                "typology": "total",
                "rings": int(summary["rings"].sum()),
                "transactions": int(summary["transactions"].sum()),
                "median_ring_size": float("nan"),
            }
        ]
    )
    return pd.concat([summary, total], ignore_index=True)
