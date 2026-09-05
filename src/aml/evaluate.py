"""Scoring a ranked list against the rules fixed in config before any fitting.

Two questions are asked of every model and they are not the same question.

At transaction level: of the laundering transactions in the test window, how
many sit inside the alert budget. This is what a per transaction system is
usually judged on.

At ring level: of the rings with any transaction in the test window, how many
have at least one, at least two, or at least half of their transactions inside
the budget. A ring caught once is a ring an investigator can pull the thread
on, which is why at_least_1 is the primary definition, and the stricter two are
reported beside it so that a model which catches a single leg of every ring is
not confused with one that surfaces the structure.

Rings with no transaction in the test window are not evaluable and are excluded
from the denominator rather than counted as missed. Their count is reported.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .config import (ALERT_BUDGETS, BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL,
                     RING_CAUGHT_SENSITIVITY, rng)
from .periods import Periods, table_for


def patterns_table(split: str) -> str:
    return f"patterns_{split.replace('-', '_').lower()}"


def ring_membership(con: duckdb.DuckDBPyConnection, split: str,
                    periods: Periods) -> pd.DataFrame:
    """Every ring transaction that falls in the test window, with its ring id.

    The join is the composite key the project verified at 100 percent in phase
    one: timestamp, both banks, both accounts and both amounts. There is no
    transaction id in the source, so this is the only way back from a pattern
    row to the transaction it describes.
    """
    trans, pat = table_for(split), patterns_table(split)
    return con.execute(
        f"""
        SELECT p.ring_id, p.typology, t.txn_id
        FROM {pat} p
        JOIN {trans} t
          ON p.ts = t.ts
         AND p.from_bank = t.from_bank
         AND p.from_account = t.from_account
         AND p.to_bank = t.to_bank
         AND p.to_account = t.to_account
         AND p.amount_paid = t.amount_paid
         AND p.amount_received = t.amount_received
        WHERE t.ts >= TIMESTAMP '{periods.cut}'
          AND t.ts <  TIMESTAMP '{periods.dense_end}'
        """
    ).df()


def _bootstrap_interval(hits: np.ndarray, label: str) -> tuple[float, float]:
    """Percentile interval over resampled units, at the configured level."""
    if hits.size == 0:
        return (float("nan"), float("nan"))
    generator = rng("bootstrap", label)
    draws = generator.integers(0, hits.size, size=(BOOTSTRAP_RESAMPLES, hits.size))
    means = hits[draws].mean(axis=1)
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return tuple(np.quantile(means, [tail, 1.0 - tail]))


def evaluate(scored, rings: pd.DataFrame, total_rings: int) -> pd.DataFrame:
    """Transaction and ring level recall at every configured alert budget."""
    order = np.argsort(-scored.score, kind="stable")
    ranked_ids = scored.txn_id[order]
    ranked_labels = scored.label[order]
    n_test = scored.txn_id.size

    evaluable = rings["ring_id"].nunique()
    ring_sizes = rings.groupby("ring_id").size()

    rows = []
    for budget in ALERT_BUDGETS:
        k = max(int(round(budget * n_test)), 1)
        alerted = set(ranked_ids[:k].tolist())

        caught_in_alert = ranked_labels[:k].sum()
        total_laundering = int(scored.label.sum())
        txn_hits = ranked_labels[:k]

        flagged_per_ring = (
            rings.assign(flagged=rings["txn_id"].isin(alerted))
            .groupby("ring_id")["flagged"].sum()
        )

        row = {
            "split": scored.split,
            "variant": scored.variant,
            "alert_budget": budget,
            "alerts": k,
            "test_transactions": n_test,
            "test_laundering": total_laundering,
            "laundering_in_alerts": int(caught_in_alert),
            "txn_recall": float(caught_in_alert) / total_laundering if total_laundering else float("nan"),
            "precision": float(caught_in_alert) / k,
            "rings_evaluable": int(evaluable),
            "rings_not_evaluable": int(total_rings - evaluable),
        }
        lo, hi = _bootstrap_interval(txn_hits.astype(float), f"{scored.split}:{scored.variant}:{budget}:txn")
        row["precision_lo"], row["precision_hi"] = lo, hi

        for definition in RING_CAUGHT_SENSITIVITY:
            if definition == "at_least_1":
                caught = flagged_per_ring >= 1
            elif definition == "at_least_2":
                caught = flagged_per_ring >= 2
            else:
                caught = flagged_per_ring >= (ring_sizes.reindex(flagged_per_ring.index) / 2.0)
            hits = caught.to_numpy().astype(float)
            row[f"ring_recall_{definition}"] = float(hits.mean()) if hits.size else float("nan")
            lo, hi = _bootstrap_interval(hits, f"{scored.split}:{scored.variant}:{budget}:{definition}")
            row[f"ring_recall_{definition}_lo"] = lo
            row[f"ring_recall_{definition}_hi"] = hi

        rows.append(row)
    return pd.DataFrame(rows)


def per_typology(scored, rings: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Ring recall broken out by typology at one budget.

    Reported because the typologies are different shapes and a single ring
    recall number hides which structures a detector can see.
    """
    order = np.argsort(-scored.score, kind="stable")
    k = max(int(round(budget * scored.txn_id.size)), 1)
    alerted = set(scored.txn_id[order][:k].tolist())

    flagged = (
        rings.assign(flagged=rings["txn_id"].isin(alerted))
        .groupby(["typology", "ring_id"])["flagged"].sum()
        .reset_index()
    )
    out = (
        flagged.assign(caught=flagged["flagged"] >= 1)
        .groupby("typology")
        .agg(rings=("ring_id", "nunique"), caught=("caught", "sum"))
        .reset_index()
    )
    out["ring_recall"] = out["caught"] / out["rings"]
    out.insert(0, "variant", scored.variant)
    out.insert(0, "split", scored.split)
    out["alert_budget"] = budget
    return out
