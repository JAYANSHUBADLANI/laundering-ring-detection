"""Locate the raw Kaggle files, downloading them only if they are absent.

The raw data is licensed CDLA-Sharing-1.0 and is never committed. This module
prefers files already sitting in data/raw so the project works for someone who
downloaded them through a browser, and falls back to the Kaggle CLI per file
rather than pulling the whole dataset, which bundles the 16 GB Large splits.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_RAW, KAGGLE_DATASET, SPLITS


class RawDataMissing(RuntimeError):
    """Raised when a required raw file is absent and cannot be downloaded."""


@dataclass(frozen=True)
class FileStatus:
    """Presence and size of one expected raw file."""

    split: str
    filename: str
    present: bool
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_048_576


def kaggle_available() -> bool:
    """True when the Kaggle CLI is installed and credentials are readable."""
    if shutil.which("kaggle") is None:
        return False
    home_token = Path.home() / ".kaggle" / "kaggle.json"
    config_token = Path.home() / ".config" / "kaggle" / "kaggle.json"
    return home_token.exists() or config_token.exists()


def status(splits: tuple[str, ...], raw_dir: Path = DATA_RAW) -> list[FileStatus]:
    """Report which expected files are present in the raw directory."""
    rows: list[FileStatus] = []
    for split_name in splits:
        for filename in SPLITS[split_name].files:
            path = raw_dir / filename
            exists = path.exists() and path.stat().st_size > 0
            rows.append(
                FileStatus(
                    split=split_name,
                    filename=filename,
                    present=exists,
                    size_bytes=path.stat().st_size if exists else 0,
                )
            )
    return rows


def download_file(filename: str, raw_dir: Path = DATA_RAW) -> None:
    """Download a single file from the dataset with the Kaggle CLI."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-f",
        filename,
        "--unzip",
        "-p",
        str(raw_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RawDataMissing(
            f"kaggle download failed for {filename}: {result.stderr.strip() or result.stdout.strip()}"
        )


def ensure(splits: tuple[str, ...], raw_dir: Path = DATA_RAW) -> list[FileStatus]:
    """Make sure every file for the given splits is present, downloading if needed."""
    missing = [row for row in status(splits, raw_dir) if not row.present]
    if missing and not kaggle_available():
        names = "\n  ".join(row.filename for row in missing)
        raise RawDataMissing(
            "the following raw files are missing and the Kaggle CLI is not available:\n  "
            f"{names}\n"
            "Either install and authenticate the Kaggle CLI, or download these files from\n"
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            "and place them in data/raw/ keeping the original filenames."
        )
    for row in missing:
        download_file(row.filename, raw_dir)

    still_missing = [row.filename for row in status(splits, raw_dir) if not row.present]
    if still_missing:
        raise RawDataMissing(f"still missing after download: {', '.join(still_missing)}")
    return status(splits, raw_dir)
