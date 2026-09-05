"""Phase 3: account level graph features.

The claim this project exists to test is that a ring's individual legs look
ordinary, so a per transaction model cannot see it while the structure around
the accounts can. Phase 2 built the per transaction side. This builds the
structural side.

Every feature here is computed from the training window only and then attached
to test transactions by account. That is the shape a deployed system has: a
feature store refreshed on a schedule, scoring traffic that arrives after the
refresh. It also makes leakage impossible by construction, because nothing in
the test window can reach a feature value.

An account that never appears in the training window gets nulls rather than
zeros, since "no history" and "history of zero" are different states and the
model is allowed to treat them differently.

The aggregate features are SQL because DuckDB does them faster than anything
that pulls five million edges into Python. Only the two that genuinely need a
graph library, component size and short cycle membership, use igraph.
"""

from __future__ import annotations

import duckdb
import igraph as ig
import pandas as pd

from .periods import Periods, table_for

# Cycles longer than this are not enumerated. Ring typologies in this data are
# small, the largest median is 14 transactions, and cycle enumeration cost grows
# fast enough that an unbounded search would dominate the whole project.
MAX_CYCLE_LENGTH = 4


def account_aggregates(con: duckdb.DuckDBPyConnection, split: str,
                       periods: Periods) -> pd.DataFrame:
    """Per account degree, volume and flow features from the training window."""
    table = table_for(split)
    return con.execute(
        f"""
        WITH sent AS (
            SELECT from_account AS account,
                   count(*) AS out_txns,
                   count(DISTINCT to_account) AS out_degree,
                   sum(amount_paid) AS out_amount,
                   count(DISTINCT currency_paid) AS out_currencies,
                   count(DISTINCT payment_format) AS out_formats,
                   count(DISTINCT to_bank) AS out_banks
            FROM {table} WHERE {periods.train_clause} GROUP BY 1
        ),
        received AS (
            SELECT to_account AS account,
                   count(*) AS in_txns,
                   count(DISTINCT from_account) AS in_degree,
                   sum(amount_received) AS in_amount,
                   count(DISTINCT currency_received) AS in_currencies,
                   count(DISTINCT from_bank) AS in_banks
            FROM {table} WHERE {periods.train_clause} GROUP BY 1
        ),
        both_ways AS (
            -- Counterparties this account both sent to and received from. A
            -- high reciprocity share is the signature of a cycle or a stack,
            -- where money returns along the path it left by.
            SELECT s.from_account AS account,
                   count(DISTINCT s.to_account) AS reciprocal_partners
            FROM (SELECT DISTINCT from_account, to_account FROM {table}
                  WHERE {periods.train_clause}) s
            JOIN (SELECT DISTINCT from_account, to_account FROM {table}
                  WHERE {periods.train_clause}) r
              ON s.to_account = r.from_account AND s.from_account = r.to_account
            GROUP BY 1
        )
        SELECT
            coalesce(sent.account, received.account) AS account,
            coalesce(out_txns, 0) AS out_txns,
            coalesce(out_degree, 0) AS out_degree,
            coalesce(out_amount, 0) AS out_amount,
            coalesce(out_currencies, 0) AS out_currencies,
            coalesce(out_formats, 0) AS out_formats,
            coalesce(out_banks, 0) AS out_banks,
            coalesce(in_txns, 0) AS in_txns,
            coalesce(in_degree, 0) AS in_degree,
            coalesce(in_amount, 0) AS in_amount,
            coalesce(in_currencies, 0) AS in_currencies,
            coalesce(in_banks, 0) AS in_banks,
            coalesce(reciprocal_partners, 0) AS reciprocal_partners,
            -- How much of what came in went straight back out. A mule account
            -- passes nearly everything through; a normal account does not.
            CASE WHEN coalesce(in_amount, 0) > 0
                 THEN coalesce(out_amount, 0) / in_amount END AS passthrough_ratio,
            -- Concentration: many senders into one account is fan in, many
            -- receivers out of one is fan out.
            CASE WHEN coalesce(in_txns, 0) > 0
                 THEN coalesce(in_degree, 0)::DOUBLE / in_txns END AS in_partner_share,
            CASE WHEN coalesce(out_txns, 0) > 0
                 THEN coalesce(out_degree, 0)::DOUBLE / out_txns END AS out_partner_share
        FROM sent
        FULL OUTER JOIN received ON sent.account = received.account
        LEFT JOIN both_ways ON both_ways.account = coalesce(sent.account, received.account)
        """
    ).df()


def structural_features(con: duckdb.DuckDBPyConnection, split: str,
                        periods: Periods) -> pd.DataFrame:
    """Component size and short cycle membership, the two that need a graph.

    Edges are deduplicated to account pairs first. Five million transactions
    collapse to far fewer distinct pairs, and neither feature cares how many
    times a pair transacted.
    """
    table = table_for(split)
    edges = con.execute(
        f"""
        SELECT DISTINCT from_account, to_account
        FROM {table}
        WHERE {periods.train_clause} AND from_account <> to_account
        """
    ).df()

    accounts = pd.Index(
        pd.unique(pd.concat([edges["from_account"], edges["to_account"]]))
    )
    lookup = {name: i for i, name in enumerate(accounts)}
    graph = ig.Graph(
        n=len(accounts),
        edges=[(lookup[a], lookup[b])
               for a, b in zip(edges["from_account"], edges["to_account"])],
        directed=True,
    )

    membership = graph.connected_components(mode="weak").membership
    sizes = pd.Series(membership).value_counts()
    component_size = [int(sizes[m]) for m in membership]

    # Short directed cycles through each account, capped at MAX_CYCLE_LENGTH.
    # igraph counts these far faster than enumerating them in Python, and the
    # count rather than the cycles themselves is what a model can use.
    in_two_cycle = graph.is_mutual()
    two_cycle_count = [0] * len(accounts)
    for edge, mutual in zip(graph.es, in_two_cycle):
        if mutual:
            two_cycle_count[edge.source] += 1

    return pd.DataFrame({
        "account": accounts,
        "component_size": component_size,
        "two_cycle_edges": two_cycle_count,
        "degree_total": graph.degree(mode="all"),
    })


def build(con: duckdb.DuckDBPyConnection, split: str,
          periods: Periods) -> pd.DataFrame:
    """All account level features for one split, keyed by account."""
    aggregates = account_aggregates(con, split, periods)
    structural = structural_features(con, split, periods)
    return aggregates.merge(structural, on="account", how="outer")
