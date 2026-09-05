"""Paths, split definitions and the single root seed for the project.

All paths are derived from the repository root at import time so that nothing
absolute is ever written into a committed file, figure or result table.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
TESTS_FIXTURES = REPO_ROOT / "tests" / "fixtures"

DUCKDB_PATH = DATA_INTERIM / "aml.duckdb"

ROOT_SEED = 20260904

KAGGLE_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


@dataclass(frozen=True)
class Split:
    """One HI or LI split of the IBM AML data."""

    name: str
    size: str
    illicit_ratio: str

    @property
    def trans_file(self) -> str:
        return f"{self.name}_Trans.csv"

    @property
    def accounts_file(self) -> str:
        return f"{self.name}_accounts.csv"

    @property
    def patterns_file(self) -> str:
        return f"{self.name}_Patterns.txt"

    @property
    def files(self) -> tuple[str, str, str]:
        return (self.trans_file, self.accounts_file, self.patterns_file)


SPLITS: dict[str, Split] = {
    "HI-Small": Split("HI-Small", "Small", "high"),
    "LI-Small": Split("LI-Small", "Small", "low"),
    "HI-Medium": Split("HI-Medium", "Medium", "high"),
    "LI-Medium": Split("LI-Medium", "Medium", "low"),
    "HI-Large": Split("HI-Large", "Large", "high"),
    "LI-Large": Split("LI-Large", "Large", "low"),
}

ACTIVE_SPLITS: tuple[str, ...] = ("HI-Small", "LI-Small")

TYPOLOGIES: tuple[str, ...] = (
    "BIPARTITE",
    "CYCLE",
    "FAN-IN",
    "FAN-OUT",
    "GATHER-SCATTER",
    "RANDOM",
    "SCATTER-GATHER",
    "STACK",
)

PATTERN_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "currency_received",
    "amount_paid",
    "currency_paid",
    "payment_format",
    "is_laundering",
)


def seed_sequence(*labels: str) -> np.random.SeedSequence:
    """Derive an independent SeedSequence from the root seed and string labels.

    Every random stream in the project spawns from ROOT_SEED through this
    function so that two consecutive runs produce identical result tables.
    The label digest uses blake2b rather than the builtin hash, which is
    salted per process and would break run to run reproducibility.
    """
    spawn_key = tuple(
        int.from_bytes(hashlib.blake2b(label.encode("utf-8"), digest_size=4).digest(), "big")
        for label in labels
    )
    return np.random.SeedSequence(entropy=ROOT_SEED, spawn_key=spawn_key)


def rng(*labels: str) -> np.random.Generator:
    """Return a Generator for a named random stream."""
    return np.random.default_rng(seed_sequence(*labels))


"""Evaluation rules, fixed before any model was fitted.

These are written down here rather than chosen later because a ring level
recall figure depends entirely on how many of a ring's transactions must be
flagged, and picking that rule after seeing results would invalidate the whole
comparison.
"""

RING_CAUGHT_MIN_FLAGGED = 1

RING_CAUGHT_SENSITIVITY: tuple[str, ...] = ("at_least_1", "at_least_2", "at_least_half")

ALERT_BUDGETS: tuple[float, ...] = (0.001, 0.005, 0.01)

PRIMARY_ALERT_BUDGET = 0.001

TEMPORAL_CUT_QUANTILE = 0.70

BOOTSTRAP_RESAMPLES = 2000

CONFIDENCE_LEVEL = 0.95


"""Period handling, decided before any model was fitted.

Both Small splits carry a hard structural break: background traffic stops and
what remains is almost entirely laundering. In HI-Small the first ten days hold
5,077,237 transactions at a 0.089 percent laundering rate and the remaining
eight days hold 1,108 at 59 percent. LI-Small breaks on the same day.

A temporal split that put that tail in the test set would hand any model a
region separable by date alone and inflate every number in the project. The
tail is not deleted either, because rings straddle it: it is held out of the
primary evaluation window and reported separately.

The boundary is derived rather than hardcoded to a date, so the same rule
carries to the Medium and Large splits without a second decision. A day belongs
to the dense period if it carries at least DENSE_DAY_MIN_SHARE of the median
daily volume. On both Small splits the margin is two orders of magnitude, 396
transactions against a 4,820 threshold in HI-Small, so the rule is not
sensitive to the exact share.
"""

DENSE_DAY_MIN_SHARE = 0.01
