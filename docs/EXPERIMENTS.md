# EntropyShop — Experiments

Every number below comes from the **unmodified** official evaluator
(`evaluator/local_evaluator.py`) over all 200 public sessions, unless the row
is explicitly labelled a paraphrase stress test.

Reproduce any row with:

```bash
python3 scripts/run_eval.py --label <name> --set section.field=value
python3 scripts/compare_policies.py          # full ablation grid
python3 scripts/tune_config.py --grid final  # weight sweep
```

## Baseline reproduction

The official weak BM25 baseline was reproduced exactly before any code was
written, confirming the harness behaves as documented:

| Metric | `docs/baseline_results.json` | Reproduced |
|---|---|---|
| Hit Rate@10 | 0.125 | 0.125 |
| MRR | 0.068034 | 0.068034 |
| MTTC | 9.81 | 9.81 |
| TechnicalScore | 0.10671 | 0.10671 |

```bash
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl \
    --dataset data/public_set.jsonl --output results.json
```

## Headroom study (before implementation)

A deliberately crude probe — category bucket filter plus phrase containment,
no ranking model — established what the *information* was worth, independent of
engineering quality:

| Requirements known | Hit Rate@10 | MRR |
|---|---|---|
| 1 | 0.325 | 0.150 |
| 2 | 0.675 | 0.528 |
| 4 | 0.935 | 0.822 |

This is why the design spends turns on elicitation rather than on ranking
sophistication.

## Ablation grid

Sorted by TechnicalScore. Rows 1–4 build the ranker up; rows 5–8 compare
question policies; rows 9–11 ablate priors and exploration; rows 12+ sweep the
recommendation budget.

| # | Configuration | Hit@10 | MRR | MTTC | Eff | **Score** |
|---|---|---|---|---|---|---|
| 1 | Official BM25 baseline | 0.125 | 0.068 | 9.81 | 0.119 | 0.10671 |
| 2 | Lexical retrieval only | 0.095 | 0.073 | 10.23 | 0.076 | 0.08458 |
| 3 | + category route | 0.480 | 0.257 | 6.49 | 0.451 | 0.40718 |
| 4 | + facet scoring | 0.945 | 0.594 | 2.48 | 0.853 | 0.82108 |
| 5 | + phrase containment (full ranker) | 0.985 | 0.637 | 2.06 | 0.894 | 0.86237 |
| 6 | Fixed question order | 1.000 | 0.616 | 2.00 | 0.900 | 0.86491 |
| 7 | Entropy policy | 1.000 | 0.641 | 1.91 | 0.909 | 0.87401 |
| 8 | Hybrid policy | 1.000 | 0.642 | 1.90 | 0.910 | 0.87458 |
| 9 | `other_first` policy | 1.000 | 0.652 | 1.90 | 0.910 | 0.87766 |
| 10 | Hybrid, no exploration | 0.985 | 0.638 | 2.04 | 0.896 | 0.86304 |
| 11 | Hybrid, no popularity prior | 0.995 | 0.662 | 2.19 | 0.880 | 0.87206 |
| 12 | Hybrid, **no profile prior** | 1.000 | 0.650 | 1.61 | 0.939 | 0.88288 |
| 13 | budget `(5,10)` | 1.000 | 0.732 | 2.03 | 0.897 | 0.89907 |
| 14 | budget `(3,10)` | 1.000 | 0.762 | 2.08 | 0.892 | 0.90711 |
| 15 | budget `(2,10)` | 1.000 | 0.783 | 2.14 | 0.886 | 0.91195 |
| 16 | budget `(1,10)` | 1.000 | 0.813 | 2.21 | 0.878 | 0.91975 |
| 17 | budget `(3,3,10)` | 1.000 | 0.800 | 2.19 | 0.880 | 0.91604 |
| 18 | budget `(2,3,5,10)` | 1.000 | 0.830 | 2.26 | 0.874 | 0.92374 |
| 19 | budget `(1,3,10)` | 1.000 | 0.865 | 2.33 | — | 0.93299 |
| 20 | budget `(1,1,2,3,5,10)` | 1.000 | 0.909 | 2.54 | — | 0.94185 |
| 21 | budget `(1,2,3,10)` | 1.000 | 0.910 | 2.44 | — | 0.94412 |
| 22 | budget `(1,2,3,5,10)` | 1.000 | 0.921 | 2.46 | — | 0.94726 |
| 23 | 22 + no profile prior | 1.000 | 0.941 | 2.18 | 0.882 | 0.95878 |
| 24 | 23 + `other_first` | 1.000 | 0.941 | 2.17 | — | 0.95908 |
| **25** | **Final default (23 + hardened parser)** | **1.000** | **0.932** | **2.19** | **0.881** | **0.95583** |

Rows 1–24 predate the parser hardening described below. Row 25 is the shipped
configuration.

### What the grid actually taught us

**Category is the dominant lever.** Row 2 → row 3 is +0.32 TechnicalScore from
one signal. Everything else combined adds less. This is also why
`coarse_category` was verified for exact parity against the harness across all
50,000 products — an early version of ours excluded a bare `Jewelry` component
and silently produced different bucket keys.

**The aggregate profile hurts** (row 8 → row 12, +0.008; and +0.02 at the
tuned budget). Its tags — `fit`, `comfort`, `durability` — match generic
apparel copy almost everywhere, so despite a weight of only 0.02 it acts as a
near-random tie-breaker among otherwise-equal candidates. It is **off by
default**, which is a measured decision, not an oversight. `w_profile` remains
in the config for anyone who wants to revisit it.

**Adaptive questioning beats a fixed order** but only modestly (0.86491 →
0.87458). The larger share of the benefit comes from *not wasting* turns —
answerability suppressing `brand`/`category` — rather than from optimal
ordering.

**Exploration matters more than it looks** (row 10 → row 8): +0.012, and it is
what lifts Hit Rate from 0.985 to 1.000.

**The budget ramp is the biggest single tuned gain**: +0.075 from row 12 to
row 23, all of it MRR. See `ARCHITECTURE.md` §2 for the reasoning.

### Policy choice: `hybrid` over `other_first`

`other_first` scores 0.95908 against `hybrid`'s 0.95878 — a gap of 0.0003,
which is roughly one session out of 200 and well inside noise. We ship
`hybrid` because it degrades gracefully: it stops asking broad questions the
moment one stops paying off and falls back to entropy-selected typed
attributes, whereas `other_first` hard-codes its opening move. Selecting
`other_first` here would be fitting to noise while reducing robustness.

## Error analysis (final configuration)

```bash
python3 scripts/inspect_errors.py
```

| Category | Count | Share |
|---|---|---|
| Hit at rank 1 | 178 | 89.0% |
| Hit at rank 2–3 | 20 | 10.0% |
| Hit at rank 4–10 | 2 | 1.0% |
| Any miss | 0 | 0.0% |

Non-rank-1 hits by scenario: browsing 9, intent_override 7, buying 5,
boundary 1. By turn: 10 at turn 2, 6 at turn 3, 3 at turn 4, 3 later.

The residual loss is concentrated in sessions where the target's metadata is
sparse, so quoted requirements match many catalog siblings equally well and
the tie is broken by priors rather than evidence.

## Robustness: the paraphrase stress test

This is the most important experiment here, and it changed the shipped code.

The competition specification warns that the organizer may add natural-language
paraphrasing to the private simulator. A parser tuned to exact template strings
would score well on the public set and collapse on the private one.
`scripts/stress_paraphrase.py` re-implements the **documented session protocol**
with paraphrased wording and identical scoring rules. Its template mode
reproduces the official score exactly, which validates the harness itself.

| Wording | Before hardening | After hardening |
|---|---|---|
| Template (control) | 0.95878 | 0.95583 |
| **Paraphrased** | **0.74187** | **0.92838** |
| Gap | **0.217** | **0.027** |

Before hardening, paraphrased buying Hit Rate fell to 0.8125. The diagnosis was
specific:

- `"Hoping to find X — must-have is Y"` — `find` was not a category trigger, so
  **no category was extracted at all**, discarding the strongest signal
- `"What I really care about: A; B"` — the word `really` broke the disclosure regex
- `"Nothing further on material"` — not recognised as an exhausted attribute

The fix was structural rather than more regexes. Parsing is now **turn-aware**:
turn 1 is always the opening, every later turn is a reply to a question we just
asked, so anything not recognised as a refusal, an override or a nudge is read
as a disclosure with its framing stripped. The default inverted from "parse if
recognised" to "information unless proven otherwise".

Cost on the public set: −0.003 (one session). Benefit under paraphrase: +0.187.
That trade is worth taking, and we would take it again even if the public
number had dropped further.

## Generalisation risks

Stated plainly, because the public-set numbers look better than they should be
trusted to be.

1. **Zero misses on 200 sessions is not a promise of zero misses on 800.**
   Report this as "no misses on the public set". The private split uses
   different users and targets.

2. **The `(1,2,3,5,10)` budget is a bet on high Hit Rate.** It presents 1
   product on turn 1 and 21 across the first five turns, versus 50 for a flat
   budget. If private Hit Rate is materially below 1.0, this ramp converts
   would-be hits into misses, and Hit Rate carries 0.50 weight against MRR's
   0.30. Mitigation: `policy.turn_budget=(3,5,10)` is measured and available
   (`scripts/tune_config.py --grid robust`), and `(10,)` restores flat
   behaviour. This is the first dial to turn if private Hit Rate disappoints.

3. **The customer model may not transfer.** Question value assumes requirements
   are short phrases quoted from product metadata, surfacing material and
   colour first. If the private simulator discloses differently, question
   *selection* degrades — but ranking does not, because it scores whatever
   text actually arrives. The agent also adapts within a session: an attribute
   that returns nothing is marked exhausted regardless of what the model
   predicted.

4. **Category resolution is tuned to a known granularity.** Exact bucket
   matching is verified against all 50,000 products, with case-insensitive and
   Jaccard-overlap fallbacks. A substantially different phrasing of the opening
   line would fall back to the lexical route, which is weaker but never empty.

5. **We do not tune on target identity.** Every swept parameter is a single
   global number. No public ASIN, sample ID or answer sequence appears in
   `starter/` — enforced by an AST-level test
   (`test_no_hardcoded_catalog_identifiers`, `test_runtime_never_imports_evaluator_or_reads_labels`).

## Resource usage

| Metric | Value |
|---|---|
| Catalog init | 11.2 s (once per process) |
| Per-turn latency | 41.95 ms mean, 101.26 ms p95 |
| Full 200-session evaluation | ~31 s including init |
| Memory | ~360 MB resident |
| Prompt tokens | 0 |
| Completion tokens | 0 |
| Estimated model cost | $0.00 |
| Network access | none required |
