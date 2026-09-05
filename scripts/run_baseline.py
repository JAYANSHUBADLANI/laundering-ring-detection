"""Entry point: phase 2, the transaction level baseline and its evaluation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import pandas as pd

from aml import baseline, evaluate, figures, graphfeat, periods
from aml.config import ACTIVE_SPLITS, DUCKDB_PATH, PRIMARY_ALERT_BUDGET, RESULTS


def main() -> int:
    pd.set_option("display.width", 200)
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    period_rows, eval_frames, typology_frames = [], [], []
    for split in ACTIVE_SPLITS:
        p = periods.compute(con, split)
        table = periods.table_for(split)
        counts = {}
        for name, clause in (("train", p.train_clause), ("test", p.test_clause),
                             ("tail", p.tail_clause)):
            n, lab = con.execute(
                f"SELECT count(*), sum(is_laundering) FROM {table} WHERE {clause}"
            ).fetchone()
            counts[name] = (n, int(lab or 0))
        period_rows.append({
            "split": split,
            "dense_end": p.dense_end,
            "temporal_cut": p.cut,
            "last_ts": p.last_ts,
            "train_txns": counts["train"][0], "train_laundering": counts["train"][1],
            "test_txns": counts["test"][0], "test_laundering": counts["test"][1],
            "tail_txns": counts["tail"][0], "tail_laundering": counts["tail"][1],
        })

        total_rings = con.execute(
            f"SELECT count(DISTINCT ring_id) FROM {evaluate.patterns_table(split)}"
        ).fetchone()[0]
        rings = evaluate.ring_membership(con, split, p)

        start = time.perf_counter()
        gstart = time.perf_counter()
        graph_features = graphfeat.build(con, split, p)
        print(f"{split:10s} graph features   {len(graph_features):>9,} accounts "
              f"in {time.perf_counter() - gstart:.1f}s")

        for scored in baseline.run_split(con, split, p, graph_features):
            eval_frames.append(evaluate.evaluate(scored, rings, total_rings))
            typology_frames.append(
                evaluate.per_typology(scored, rings, PRIMARY_ALERT_BUDGET))
            print(f"{split:10s} {scored.variant:24s} "
                  f"train {scored.train_rows:>9,}  test {scored.test_rows:>9,}")
        print(f"{split:10s} fitted both variants in {time.perf_counter() - start:.1f}s")

    pd.DataFrame(period_rows).to_csv(RESULTS / "periods.csv", index=False)
    results = pd.concat(eval_frames, ignore_index=True)
    results.to_csv(RESULTS / "baseline_evaluation.csv", index=False)
    pd.concat(typology_frames, ignore_index=True).to_csv(
        RESULTS / "baseline_by_typology.csv", index=False)

    figure = figures.detection_figure(results, PRIMARY_ALERT_BUDGET)
    print(f"wrote {figure}")

    show = ["split", "variant", "alert_budget", "alerts", "txn_recall", "precision",
            "ring_recall_at_least_1", "ring_recall_at_least_2", "rings_evaluable"]
    print("\n" + results[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
