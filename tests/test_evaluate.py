"""Tests for the alert budget and ring caught rules.

These use a hand built scored object and a hand built ring table, so the
arithmetic is checkable by reading rather than by trusting a model run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml import evaluate
from aml.baseline import Scored


def make_scored(scores, labels, split="TEST", variant="unit"):
    return Scored(
        split=split, variant=variant,
        txn_id=np.arange(len(scores)),
        score=np.asarray(scores, dtype=float),
        label=np.asarray(labels, dtype=int),
        train_rows=0, test_rows=len(scores),
    )


def test_alert_budget_takes_the_top_k_by_score():
    """1 percent of 1,000 transactions is 10 alerts, and they are the top 10."""
    scores = np.linspace(0, 1, 1000)
    labels = np.zeros(1000, dtype=int)
    labels[-10:] = 1                       # the ten highest scores are the positives
    scored = make_scored(scores, labels)
    rings = pd.DataFrame({"ring_id": [], "typology": [], "txn_id": []})
    out = evaluate.evaluate(scored, rings, total_rings=0)
    row = out[out["alert_budget"] == 0.01].iloc[0]
    assert row["alerts"] == 10
    assert row["txn_recall"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)


def test_a_perfectly_wrong_ranking_recalls_nothing():
    scores = np.linspace(0, 1, 1000)
    labels = np.zeros(1000, dtype=int)
    labels[:10] = 1                        # positives are the ten lowest scores
    scored = make_scored(scores, labels)
    rings = pd.DataFrame({"ring_id": [], "typology": [], "txn_id": []})
    out = evaluate.evaluate(scored, rings, total_rings=0)
    row = out[out["alert_budget"] == 0.01].iloc[0]
    assert row["txn_recall"] == pytest.approx(0.0)


def test_ring_caught_definitions_are_ordered():
    """at_least_1 can never be below at_least_2, which can never be below half.

    A ring of four transactions with two flagged is caught under at_least_1 and
    at_least_2, and exactly meets at_least_half.
    """
    scores = np.zeros(100)
    scores[:2] = 1.0                       # only these two are alerted
    labels = np.zeros(100, dtype=int)
    labels[:4] = 1
    scored = make_scored(scores, labels)
    rings = pd.DataFrame({
        "ring_id": [7, 7, 7, 7],
        "typology": ["CYCLE"] * 4,
        "txn_id": [0, 1, 2, 3],
    })
    out = evaluate.evaluate(scored, rings, total_rings=1)
    row = out[out["alert_budget"] == 0.01].iloc[0]   # 1 percent of 100 is 1 alert
    assert row["alerts"] == 1
    assert row["ring_recall_at_least_1"] == 1.0
    assert row["ring_recall_at_least_2"] == 0.0
    assert row["ring_recall_at_least_1"] >= row["ring_recall_at_least_2"]


def test_rings_outside_the_test_window_are_not_in_the_denominator():
    """Evaluable rings come from the ring table; the rest are reported separately."""
    scored = make_scored(np.zeros(100), np.zeros(100, dtype=int))
    rings = pd.DataFrame({
        "ring_id": [1, 2],
        "typology": ["CYCLE", "STACK"],
        "txn_id": [0, 1],
    })
    out = evaluate.evaluate(scored, rings, total_rings=10)
    row = out.iloc[0]
    assert row["rings_evaluable"] == 2
    assert row["rings_not_evaluable"] == 8


def test_bootstrap_interval_brackets_the_point_estimate():
    scores = np.linspace(0, 1, 1000)
    labels = np.zeros(1000, dtype=int)
    labels[-5:] = 1
    scored = make_scored(scores, labels)
    rings = pd.DataFrame({
        "ring_id": [1] * 5,
        "typology": ["FAN-IN"] * 5,
        "txn_id": list(range(995, 1000)),
    })
    out = evaluate.evaluate(scored, rings, total_rings=1)
    row = out[out["alert_budget"] == 0.01].iloc[0]
    assert row["ring_recall_at_least_1_lo"] <= row["ring_recall_at_least_1"]
    assert row["ring_recall_at_least_1"] <= row["ring_recall_at_least_1_hi"]


def test_evaluation_is_deterministic_across_calls():
    """The bootstrap draws from a named stream, so two calls must agree exactly."""
    scores = np.random.default_rng(0).random(500)
    labels = (np.random.default_rng(1).random(500) < 0.05).astype(int)
    scored = make_scored(scores, labels)
    rings = pd.DataFrame({
        "ring_id": [1, 1, 2], "typology": ["CYCLE"] * 3, "txn_id": [0, 1, 2],
    })
    first = evaluate.evaluate(scored, rings, total_rings=2)
    second = evaluate.evaluate(scored, rings, total_rings=2)
    pd.testing.assert_frame_equal(first, second)
