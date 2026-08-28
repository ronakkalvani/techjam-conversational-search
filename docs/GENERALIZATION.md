# Generalization and Leakage Audit

## Conclusion

EntropyShop has **no direct runtime label leakage**. The agent package does not
import the evaluator, read the public session file, access ground-truth fields,
or contain public target ASINs. It uses the frozen catalog metadata that the
competition explicitly provides.

The reported `0.955831` result is nevertheless a **public development score**,
not an unbiased estimate of private performance. Ranking weights, question
policies, parser behavior, priors, and recommendation budgets were compared on
all 200 public sessions. The public evaluator also influenced the customer
model used for question-value estimation. This is legitimate development, but
it creates public-simulator overfitting risk.

## Frozen baseline

The version developed against all 200 public sessions is tagged
`public-baseline-v1`. Future changes should be compared against that tag and
follow the split protocol below.

Because every public session influenced the baseline before this protocol was
introduced, the internal test fold is **not retrospectively untouched**. It is
a guard for future modifications only. The 800 private organizer sessions
remain the only genuinely unseen official evaluation.

## Deterministic split

`data/splits/public_v1.json` assigns all 200 public sample IDs to five folds
using a seeded SHA-256 ordering within each scenario. The official source file
is never edited.

Each fold contains exactly:

| Scenario | Sessions |
|---|---:|
| Buying | 16 |
| Browsing | 16 |
| Intent Override | 6 |
| Boundary | 2 |
| **Total** | **40** |

The named splits are:

| Split | Folds | Sessions | Purpose |
|---|---|---:|---|
| Development | 1–3 | 120 | Feature work and error inspection |
| Validation | 4 | 40 | Policy, weight, and budget selection |
| Internal test | 5 | 40 | Aggregate-only check after freezing a future change |

All 200 public targets are unique. `difficulty_bucket` is perfectly aligned
with scenario in this release—Buying is easy, Browsing and Boundary are medium,
and Intent Override is hard—so scenario stratification also preserves the
difficulty distribution.

## Reproduce and verify the manifest

```bash
# Fails if the committed manifest is stale or the source dataset changed
python scripts/make_splits.py --check

# Regenerate intentionally
python scripts/make_splits.py --force
```

The manifest records the public file's SHA-256 digest, seed, scenario counts,
and explicit caveat. Validation rejects missing IDs, invented IDs, duplicate
assignment, or folds that are not partitioned exactly once.

## Evaluation workflow

```bash
# Use freely while developing
python scripts/run_split_eval.py --split development

# Use to select between otherwise finished alternatives
python scripts/run_split_eval.py --split validation

# Use once after freezing a future change; aggregate output only
python scripts/run_split_eval.py --split test

# Diagnostic stability report across all five folds
python scripts/run_split_eval.py --split folds \
  --output results/public_v1_folds.json
```

The runner refuses `--include-sessions` for `test`, `fold_5`, or any combined
selection containing the test fold. This is a workflow guard rather than a
security boundary—the public labels remain organizer-provided development data.

## Frozen-policy fold results

The unchanged `public-baseline-v1` policy produces:

| Fold | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| 1 | 1.000 | 0.912500 | 2.175 | 0.950250 |
| 2 | 1.000 | 0.933333 | 1.975 | 0.960500 |
| 3 | 1.000 | 0.933333 | 2.125 | 0.957500 |
| 4 (validation) | 1.000 | 0.965278 | 2.150 | 0.966583 |
| 5 (future test guard) | 1.000 | 0.916071 | 2.525 | 0.944321 |

Across folds, TechnicalScore has mean `0.955831`, minimum `0.944321`, maximum
`0.966583`, and sample standard deviation `0.008715`. The stable Hit Rate is
encouraging; the MRR and MTTC variation shows why the aggregate public result
should not be presented as a guaranteed private-set score.

## Remaining generalization risk

A random public-session split cannot test a different simulator because every
fold uses the same metadata-to-intent mechanism and response policy. The
existing paraphrase test measures language robustness but retains the same
targets and underlying intent cards.

The next independent test should select non-public targets from the remaining
catalog and use a separately authored customer generator with altered:

- category wording and synonym choice;
- requirement order and number of disclosed requirements;
- refusal and no-preference phrasing;
- override timing and language;
- missing or weak metadata; and
- material, color, and budget disclosure priority.

One final generated batch should be frozen before evaluation and never used for
configuration selection. Its purpose is to measure the gap from public targets
to unseen targets and then from template language to shifted language.
