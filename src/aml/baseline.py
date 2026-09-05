"""Phase 2: the transaction level baseline.

This scores every transaction on its own attributes, with no notion of the
account graph and no memory of which accounts appeared before. That is
deliberate. The claim this project exists to test is that per transaction
scoring misses rings, and a claim like that needs a per transaction model that
was given a fair chance rather than a straw man.

Two things are kept out of the feature set on purpose.

Account identifiers are not features. A model given them can memorise the
accounts that laundered in the training window and find them again in the test
window, which measures account recall rather than transaction scoring, and
would flatter this baseline for a reason that has nothing to do with reading a
transaction.

Nothing derived from the pattern file is a feature. Ring membership is the
label side of this project, not the input side.

One feature needs its own warning, and it is measured rather than argued about.
`payment_format` is close to a giveaway in this data: in HI-Small, ACH carries
4,483 of the 5,177 laundering transactions while covering 11.8 percent of rows,
and Reinvestment and Wire carry none at all. That is a property of the
generator, not of laundering. So every model here is fitted twice, once with
the format and once without, and both are reported. The version without it is
the one to believe.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import rng
from .periods import Periods, table_for

CATEGORICAL = ("payment_format", "currency_paid")

NUMERIC = (
    "log_amount_paid",
    "log_amount_received",
    "amount_ratio",
    "is_fx",
    "same_bank",
    "self_loop",
    "hour",
    "day_of_week",
    "round_100",
    "round_1000",
)


def feature_sql(table: str, clause: str) -> str:
    """One row per transaction, attributes only, no identifiers and no graph."""
    return f"""
        SELECT
            txn_id,
            ln(1 + amount_paid)                                   AS log_amount_paid,
            ln(1 + amount_received)                               AS log_amount_received,
            amount_received / nullif(amount_paid, 0)              AS amount_ratio,
            CASE WHEN currency_paid <> currency_received THEN 1 ELSE 0 END AS is_fx,
            CASE WHEN from_bank = to_bank THEN 1 ELSE 0 END       AS same_bank,
            CASE WHEN from_account = to_account THEN 1 ELSE 0 END AS self_loop,
            hour(ts)                                              AS hour,
            dayofweek(ts)                                         AS day_of_week,
            CASE WHEN amount_paid = round(amount_paid / 100) * 100 THEN 1 ELSE 0 END   AS round_100,
            CASE WHEN amount_paid = round(amount_paid / 1000) * 1000 THEN 1 ELSE 0 END AS round_1000,
            payment_format,
            currency_paid,
            is_laundering
        FROM {table}
        WHERE {clause}
    """


def load_frame(con: duckdb.DuckDBPyConnection, split: str, clause: str) -> pd.DataFrame:
    frame = con.execute(feature_sql(table_for(split), clause)).df()
    for col in CATEGORICAL:
        frame[col] = frame[col].astype("category")
    return frame


@dataclass
class Scored:
    """Test set scores from one fitted model."""

    split: str
    variant: str
    txn_id: np.ndarray
    score: np.ndarray
    label: np.ndarray
    train_rows: int
    test_rows: int


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, split: str,
                  variant: str, use_format: bool,
                  extra_numeric: list[str] | None = None) -> Scored:
    """Fit on the training window, score the test window.

    The categorical columns are aligned to the training categories before
    scoring so that a value seen only in the test window cannot shift the
    encoding of every other value.
    """
    columns = list(NUMERIC) + list(extra_numeric or []) + [
        c for c in CATEGORICAL if use_format or c != "payment_format"]
    categorical_mask = [c in CATEGORICAL for c in columns]

    x_train = train[columns].copy()
    x_test = test[columns].copy()
    for col in CATEGORICAL:
        if col in columns:
            x_test[col] = pd.Categorical(x_test[col], categories=x_train[col].cat.categories)

    model = HistGradientBoostingClassifier(
        categorical_features=categorical_mask,
        max_iter=200,
        learning_rate=0.1,
        # The label is one in a thousand, so the leaves have to be allowed to be
        # small or the model cannot separate anything at all.
        min_samples_leaf=50,
        random_state=int(rng("baseline", split, variant).integers(0, 2**31 - 1)),
    )
    model.fit(x_train, train["is_laundering"].to_numpy())
    scores = model.predict_proba(x_test)[:, 1]
    return Scored(
        split=split,
        variant=variant,
        txn_id=test["txn_id"].to_numpy(),
        score=scores,
        label=test["is_laundering"].to_numpy(),
        train_rows=len(train),
        test_rows=len(test),
    )


GRAPH_COLUMNS = (
    "out_txns", "out_degree", "out_amount", "out_currencies", "out_formats",
    "out_banks", "in_txns", "in_degree", "in_amount", "in_currencies",
    "in_banks", "reciprocal_partners", "passthrough_ratio", "in_partner_share",
    "out_partner_share", "component_size", "two_cycle_edges", "degree_total",
)


def _with_graph(con: duckdb.DuckDBPyConnection, split: str, clause: str,
                features: pd.DataFrame) -> pd.DataFrame:
    """Transaction rows plus the graph features of both accounts involved."""
    table = table_for(split)
    base = con.execute(feature_sql(table, clause)).df()
    ids = con.execute(
        f"SELECT txn_id, from_account, to_account FROM {table} WHERE {clause}"
    ).df()
    base = base.merge(ids, on="txn_id", how="left")

    sender = features.add_prefix("src_").rename(columns={"src_account": "from_account"})
    receiver = features.add_prefix("dst_").rename(columns={"dst_account": "to_account"})
    base = base.merge(sender, on="from_account", how="left")
    base = base.merge(receiver, on="to_account", how="left")
    base = base.drop(columns=["from_account", "to_account"])
    for col in CATEGORICAL:
        base[col] = base[col].astype("category")
    return base


def run_split(con: duckdb.DuckDBPyConnection, split: str,
              periods: Periods, graph_features: pd.DataFrame | None = None
              ) -> list[Scored]:
    """Every variant for one split, fitted on train and scored on test.

    The transaction only variants are the phase 2 baseline. The graph variants
    add account structure computed from the training window alone, which is the
    comparison this project was built to make.
    """
    train = load_frame(con, split, periods.train_clause)
    test = load_frame(con, split, periods.test_clause)
    scored = [
        fit_and_score(train, test, split, "txn_with_format", use_format=True),
        fit_and_score(train, test, split, "txn_without_format", use_format=False),
    ]
    if graph_features is None:
        return scored

    gtrain = _with_graph(con, split, periods.train_clause, graph_features)
    gtest = _with_graph(con, split, periods.test_clause, graph_features)
    extra = [c for c in GRAPH_COLUMNS]
    graph_cols = [f"src_{c}" for c in extra] + [f"dst_{c}" for c in extra]
    scored += [
        fit_and_score(gtrain, gtest, split, "graph_with_format",
                      use_format=True, extra_numeric=graph_cols),
        fit_and_score(gtrain, gtest, split, "graph_without_format",
                      use_format=False, extra_numeric=graph_cols),
    ]
    return scored
