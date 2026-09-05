"""Pull only the active split files out of the full dataset zip.

The Kaggle whole dataset archive carries all six splits and expands to about
39 GB. This extracts just the files for the active splits, matching on
basename so it does not care how the archive nests its directories.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.config import ACTIVE_SPLITS, DATA_RAW, SPLITS


def wanted_filenames(splits: tuple[str, ...]) -> set[str]:
    return {name for split in splits for name in SPLITS[split].files}


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--dest", type=Path, default=DATA_RAW)
    args = parser.parse_args()

    if not args.zip_path.exists():
        print(f"archive not found: {args.zip_path.name}", file=sys.stderr)
        return 1

    wanted = wanted_filenames(ACTIVE_SPLITS)
    args.dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as archive:
        members = [m for m in archive.infolist() if Path(m.filename).name in wanted]
        found = {Path(m.filename).name for m in members}
        missing = wanted - found
        if missing:
            print(f"archive does not contain: {', '.join(sorted(missing))}", file=sys.stderr)
            print("\nnames present in the archive:", file=sys.stderr)
            for info in archive.infolist():
                print(f"  {info.filename}  ({info.file_size / 1_048_576:.1f} MB)", file=sys.stderr)
            return 1

        needed = sum(m.file_size for m in members)
        available = free_bytes(args.dest)
        print(f"extracting {len(members)} files, {needed / 1_048_576:.1f} MB")
        print(f"free space at destination: {available / 1_073_741_824:.1f} GB")
        if needed > available * 0.9:
            print("not enough free space at the destination", file=sys.stderr)
            return 1

        for info in members:
            target = args.dest / Path(info.filename).name
            with archive.open(info) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            print(f"  {target.stat().st_size / 1_048_576:8.1f} MB  {target.name}")

    print(f"\nfree space after: {free_bytes(args.dest) / 1_073_741_824:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
