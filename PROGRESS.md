# Progress log

## Status

Phases 1 to 3 are complete for both Small splits. The pattern parser reproduces
the published typology counts exactly, the composite key join verifies at 100
percent, the base rates and typology tables come from the data rather than being
quoted, and both a transaction level baseline and a graph feature model have
been fitted and scored against the evaluation rules that were fixed before any
of it was run. What remains is the write up and the failure analysis, not more
modelling.

## Scope decision, taken up front

The full dataset is six splits, roughly 39 GB of CSV, which with a DuckDB
database of comparable size needs about 77 GB. The machine this is built on had
14 GB free, so the six split version is not runnable here. I scoped the work to
HI-Small and LI-Small, about 1.2 GB of CSV.

That still answers most of the question. The two splits differ mainly in base
rate, so the comparison that matters, how detection degrades as laundering gets
rarer, survives intact. What I lose is the scaling question: at which point each
graph feature stops being computable. Every stage is written split agnostic and
wall clock is recorded per stage from the start, so Medium and Large are a
config change rather than a rewrite. Until then the scaling question is out of
scope and listed as such.

## Findings so far

**The column order in the transaction file is received before paid.** The header
reads Timestamp, From Bank, Account, To Bank, Account, Amount Received,
Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is
Laundering. Any code that assumes the sent amount comes first will silently swap
the two sides of every cross currency transaction.

**Bank identifiers carry leading zeros** in both the transaction file and the
pattern file, so they are cast to integer on both sides of the join rather than
compared as text.

**The join is exact.** All 3,209 HI-Small pattern rows and all 1,023 LI-Small
pattern rows match exactly one transaction row. Zero multiple matches, zero
unmatched, under both the full key and the reduced key of timestamp, banks,
accounts and amounts. Reported in `results/join_verification.csv`.

**The pattern file does not cover all labelled laundering.** HI-Small carries
5,177 transactions with the laundering label but only 3,209 of them sit inside a
named ring, so 38 percent of labelled laundering has no named structure. In
LI-Small it is worse: 1,023 of 3,565, so 71 percent has no structure. Every ring
transaction is labelled, so the join is internally consistent, but the ring
level job can only ever speak to a subset of the label. That caveat belongs next
to every ring level number.

**There is a hard structural break on 10 September 2022.** Background traffic
stops. In HI-Small the first ten days hold 5,077,237 transactions at a 0.089
percent laundering rate, while the remaining eight days hold 1,108 transactions
at 59 percent. A temporal split that puts that tail in the test set hands the
model a region separable by date alone, and would have inflated every number in
this project. It cannot simply be deleted either, because 102 of the 370 rings
straddle the break and 7 sit entirely inside it.

The decision: rings stay defined over the full period so no ring is truncated,
the temporal cut sits inside the dense period, and the primary evaluation window
excludes the post break tail with the tail reported separately. Excluding it
removes the easiest positives, so it is the conservative direction rather than a
flattering one.

**The typologies are genuinely different shapes.** GATHER-SCATTER is the largest
at a median 14 transactions across 15 accounts over 6.3 days, RANDOM the
smallest at 3 transactions across 4 accounts over 1.9 days. Two generator
signatures are already visible: every FAN-IN ring is single currency in both
splits, and every typology has a median of exactly one payment format per ring.
Those are worth remembering when the structural detectors start scoring well,
because a detector keyed on them is reading the generator rather than
laundering.

## Base rates, measured

| split | transactions | laundering | base rate | period |
| --- | --- | --- | --- | --- |
| HI-Small | 5,078,345 | 5,177 | 0.102 percent, 1 in 981 | 1 to 18 September 2022 |
| LI-Small | 6,924,049 | 3,565 | 0.051 percent, 1 in 1,942 | 1 to 17 September 2022 |

At transaction level the two splits differ by a factor of two, not the larger
gap the ring counts alone suggest. At ring level the gap is wider: 370 rings in
5.1 million transactions against 117 in 6.9 million, a factor of 4.3.

## Evaluation rules, fixed before any model was fitted

Written into `src/aml/config.py` rather than chosen later, because ring level
recall depends entirely on how many of a ring's transactions must be flagged and
picking that rule after seeing results would invalidate the comparison.

- A ring counts as caught when at least one of its transactions inside the test
  window is flagged. Sensitivity is reported at two transactions and at half the
  ring.
- Rings with no transactions in the test window are not evaluable and are
  excluded from the denominator, with the count reported.
- The operating point is an alert budget: the top 0.1 percent of test
  transactions by score, with 0.5 percent and 1 percent also reported.
- The temporal cut is the 70th percentile of transaction time within the dense
  period.
- Intervals on ring level rates come from 2,000 bootstrap resamples at 95
  percent.

## Done

- Repository layout, MIT licence, gitignore keeping the CDLA licensed raw files
  and the DuckDB database out of version control.
- Config with paths derived from the repository root, so no absolute path can
  reach a committed file, and one root seed fanned out through
  `numpy.random.SeedSequence`. The first version of that used the builtin `hash`
  for the per stream digest, which is salted per process and would have produced
  different numbers on every run while looking deterministic. Replaced with
  blake2b and verified across three processes with `PYTHONHASHSEED=random`.
- Pattern parser that raises on a nested BEGIN, an END whose typology does not
  match its BEGIN, a transaction line outside any block, an unterminated block,
  and a wrong field count.
- Fetch module that prefers files already in `data/raw`, falls back to per file
  Kaggle CLI calls rather than the whole dataset, and a selective extractor that
  pulls only the active split files out of the full archive.
- DuckDB load for both splits, 5.1 and 6.9 million rows, 40 and 49 seconds.
- Join verification, base rates, ring label coverage, daily volume profile,
  typology description table, typology figure.
- 14 parser tests running against a committed fixture with no network call.

## Phase 2 and 3: a transaction baseline, and what the graph adds

Both are run by `scripts/run_baseline.py` into `results/baseline_evaluation.csv`,
with the per typology cut in `results/baseline_by_typology.csv` and the window
definitions in `results/periods.csv`.

**Setup.** Gradient boosting on a temporal split at the 70th percentile of the
dense period, post break tail excluded from both sides. HI-Small trains on
3,554,066 and tests on 1,523,171, LI-Small on 4,846,510 and 2,077,316. 218 of
HI-Small's 370 rings and 68 of LI-Small's 117 are evaluable. Account identifiers
are never features, and no graph feature is computed from anything later than
the training window.

**Every model fitted twice, with and without `payment_format`,** because the
generator writes a format signature: ACH carries 4,483 of HI-Small's 5,177
laundering transactions on 11.8 percent of rows.

Ring recall at the 0.1 percent budget, at least one flagged, 95 percent
bootstrap intervals:

| split | model | ring recall | interval |
|---|---|---|---|
| HI-Small | transaction, with format | 0.4404 | 0.3761 to 0.5046 |
| HI-Small | transaction, without format | 0.0459 | 0.0183 to 0.0780 |
| HI-Small | graph, with format | 0.4083 | 0.3486 to 0.4771 |
| HI-Small | graph, without format | 0.3761 | 0.3119 to 0.4404 |
| LI-Small | transaction, with format | 0.5000 | 0.3676 to 0.6176 |
| LI-Small | transaction, without format | 0.0441 | 0.0000 to 0.1029 |
| LI-Small | graph, with format | 0.3088 | 0.2059 to 0.4118 |
| LI-Small | graph, without format | 0.2353 | 0.1467 to 0.3382 |

**The thesis holds on both splits.** Without the artefact, transactions alone
find 4.6 percent of HI-Small's rings and structure finds 37.6, eight times more,
intervals not overlapping. LI-Small is the same shape lower down, 0.0441 against
0.2353, also not overlapping. The graph model barely notices whether the format
column exists, 0.4083 against 0.3761, while the transaction model lives or dies
by it, 0.4404 against 0.0459.

**A timezone bug in the temporal cut, caught by a test written after the first
run.** `periods.compute` took the quantile on epoch seconds and converted back
with `to_timestamp`, which returns a value in the session's local zone; dropping
the tzinfo kept the shifted wall clock and moved the cut five and a half hours
late on this machine. Every phase 2 and 3 number in an earlier draft of this file
was computed on that wrong cut. The quantile is now taken on the timestamp
column directly, the test asserts the cut lands at the configured quantile of
the dense period, and every number above is from the corrected run. Worth
recording because the bug was invisible in the output: the split still looked
reasonable, the models still trained, and only an assertion about where the cut
should fall exposed it.

**Per typology, HI-Small, both without the format column.** Scatter gather
0.633, fan out 0.583, gather scatter 0.500, bipartite 0.333, stack 0.320, fan in
0.240, random 0.208, cycle 0.125. The shapes the features read best are hubs,
which degree and partner concentration see directly. The worst are chains, where
every account has degree two and nothing is locally unusual. Fan in at 0.240 is
a hub shape that should be easy and is not, and I have not run that down.

## Join tests

The join every number rests on now has 25 tests of its own in
`tests/test_join.py`, built on hand made rows so the right answer is known by
reading. Each column of both keys has a test where a transaction differs from a
ring leg in that column alone and must not match, and the expected columns are
written out in the test rather than read from the module, so dropping one from
either key fails instead of shrinking the test list. Checked by deleting each
key column in turn and confirming a failure every time. 50 tests in total.

## Pending

- A failure analysis: which rings the graph model still misses at the primary
  budget and whether they share a typology or a size.
- The Medium splits, to settle whether the LI-Small result is a real failure of
  the graph features at a low base rate or simply too few evaluable rings to
  measure. Config change, several hours of load and fit.
- Standalone structural detectors, scored on their own precision and recall
  before any model sees them. The graph features are currently only inputs to a
  gradient booster, so the project cannot yet say which single structure is
  doing the work.

## Open decisions

- Cross currency amounts cannot be summed. Rings carry up to eleven distinct
  currencies, so I intend to work in within currency ranks rather than invent an
  exchange rate table, since any rate table I picked would be fiction presented
  as precision.
- Whether the transaction level job should be scored against the full laundering
  label or only against ring member transactions. The full label is the honest
  target for a per transaction system, but it includes the unstructured majority
  in LI-Small, so both will be reported.
