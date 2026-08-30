# EntropyShop Submission Pack

This file contains paste-ready Devpost copy and the recording plan for the
three-minute demonstration. The video URL must be added after uploading the
public YouTube demo.

## Project title

EntropyShop — Information-Theoretic Conversational Shopping

## Tagline

An offline shopping copilot that asks the question with the highest expected
information value and finds the customer's hidden product in fewer turns.

## Short description

Most shopping searches start with an incomplete description rather than a
product name. EntropyShop treats the conversation as a search for the missing
clues: it ranks catalog candidates, maintains uncertainty over possible
products, and asks the question most likely to make the next recommendation
better. The result is a deterministic, offline shopping copilot with a
0.955831 TechnicalScore on the 200-session public benchmark.

## Inspiration

Shopping assistants are often evaluated as if the customer already knows the
right keywords. In practice, people reveal their preferences incrementally:
first a product type, then a material, then a fit, use case, or detail that
distinguishes one item from another.

That made us focus on the interaction itself. The central design question was
not only which product to display, but which unanswered preference would be
worth learning next. Akinator was an inspiration for this question-driven
narrowing pattern, but we adapted the underlying idea to a fixed product
catalog: use each turn to reduce uncertainty about a hidden target.

The approach also fit the constraints of the challenge. With 50,000 catalog
items and a ten-turn limit, the agent must make progress quickly, avoid losing
the correct item through an overly strict filter, and ask questions that a
customer can realistically answer.

## Why a deterministic system

We considered the usual LLM-based route, including semantic retrieval and
generated responses. We chose not to put an LLM in the submitted runtime for a
deliberate reason: this task rewards faithful constraint handling and efficient
catalog search more than open-ended prose generation.

An LLM would add latency, cost, run-to-run variability, and a new failure mode:
it could paraphrase or hallucinate a product attribute that is not present in
the catalog. A model-free policy lets us inspect why a product was ranked, why
a constraint was applied, and why a particular question was selected. It also
makes the result reproducible in the organizer's local evaluation environment.

This is a task-fit decision rather than an anti-LLM claim. We use explicit
algorithms where the catalog is structured and the final action must be
auditable: lexical evidence, product facets, posterior scores, and expected
information gain.

## Our solution

EntropyShop treats the customer's intended product as a hidden target. For
every turn, it:

1. extracts constraints and intent changes from the new message;
2. retrieves candidates through category and lexical routes;
3. ranks them using phrase evidence, facet coverage, conflict penalties, and
   deterministic priors;
4. estimates a posterior over the remaining candidates; and
5. asks an information-gain question when another clue is likely to help.

The category and lexical routes are unioned rather than chained as a hard
filter. Category matching gives the system a strong signal for product type,
while lexical matching preserves useful customer wording. This protects the
target when one interpretation is incomplete.

The conversation state also handles rejected products and changing intent. An
override can clear stale suppression and reopen exploration instead of leaving
the agent trapped by an earlier preference.

## How we built it

The implementation uses Python 3.11 and the standard library. It builds
immutable in-memory indexes over the frozen 50,000-product catalog. The ranker
combines IDF-weighted lexical matching, phrase containment, catalog facets,
conflict penalties, and deterministic priors.

The question selector does not use raw entropy alone. Raw entropy can prefer a
well-partitioning attribute such as brand even when the customer cannot answer
it. Our policy combines information gain with answerability, expected ranking
improvement, missing-value probability, and turn cost.

We also implemented intent-conditioned routing and a sparse semantic route as
experimental variants. They remain available for analysis, but the final
submission uses the simpler baseline because it performed better on the
held-out guard and is easier to reproduce.

The repository includes official evaluator integration, ablation scripts,
robustness tests, and 57 automated tests.

The runtime has no external service, API key, model download, or network
dependency.

## Results and analysis

On the 200-session public development benchmark, EntropyShop substantially
improves on the organizer's weak BM25 baseline:

| Metric | Official baseline | EntropyShop |
|---|---:|---:|
| Hit Rate@10 | 0.125 | **1.000000** |
| MRR | 0.068034 | **0.932103** |
| MTTC | 9.81 | **2.190** |
| Efficiency | 0.119 | **0.881** |
| TechnicalScore | 0.10671 | **0.955831** |
| Runtime model tokens | 0 | **0** |

The score did not come from one large model or one magic feature. It came from
matching each part of the design to a failure mode we could measure:

- Lexical retrieval alone scored 0.08458. Adding the category route raised the
  score to 0.40718, showing that recognizing the product type was the first
  major bottleneck.
- Adding facet scoring raised TechnicalScore to 0.82108, because explicit
  attributes such as material and fit were more informative than generic word
  overlap.
- Phrase containment and the full ranker raised it to 0.86237.
- Adaptive questioning improved the fixed-order policy from 0.86491 to 0.87458.
  The gain came mainly from avoiding questions that customers could not answer.
- Exploration and a staged recommendation budget improved target coverage and
  MRR without sacrificing Hit Rate. The final policy reaches rank-quality
  improvements while keeping the target in the top ten for every public
  session.

The frozen policy achieved Hit Rate@10 of 1.000 on every 40-session fold. Across
the five deterministic folds, TechnicalScore ranged from 0.944321 to 0.966583,
with mean 0.955831 and standard deviation 0.008715. A separate paraphrase
stress test scored 0.92838.

The final release initializes the catalog in approximately 9.6 seconds and
averages 39.27 ms per turn, with an 84.88 ms p95 latency on the recorded
machine. These measurements are hardware-dependent, but illustrate the
practical benefit of a small in-memory deterministic system.

The public sessions influenced early development, so the public folds are
guards for later changes rather than substitutes for the organizer's unseen
private evaluation. We make this distinction explicit rather than presenting
the public score as a guarantee.

## AI assistance disclosure

ChatGPT was used during development as an engineering assistant for
brainstorming, code drafting and review, debugging, test design, evaluation
analysis, and documentation editing. The team made and reviewed the final
design and implementation decisions. No LLM is called by the submitted runtime;
the agent is deterministic, offline, and reports zero model tokens. See
`docs/AI_ASSISTANCE.md` for the full disclosure.

## Dataset and tools

The evaluation uses the organizer-provided frozen 50,000-product catalog and
200 public development sessions from the participant kit, derived from Amazon
Reviews 2023. Development tools include Python 3.11, Conda, Git, and pytest.
The runtime uses the Python standard library only; pytest is used for
development tests. No runtime APIs, external model assets, images, videos,
user identifiers, reviews, private sessions, or network services are used.

## Challenges

The hardest issue was deciding which question was worth spending a turn on.
Raw entropy prefers attributes that partition the catalog well, even when the
customer cannot answer them. We therefore made question value depend on both
uncertainty reduction and answerability.

Another challenge was avoiding overfitting to the public simulator's wording.
A paraphrase stress test exposed that an opening such as “Hoping to find…” was
not parsed as a category. We fixed the structure rather than accumulating more
templates: parsing is now turn-aware, treating the first message as an opening
and subsequent non-refusal messages as evidence.

## What we learned

The biggest lesson was that conversational recommendation is an evidence
acquisition problem as much as a ranking problem. A stronger ranker cannot
recover information the customer has not yet supplied. The right question can
be more valuable than another paragraph of generated explanation.

We also learned that the simplest signal was often the strongest one. Category
information provided the largest single improvement, while generic profile
signals sometimes acted as noisy tie-breakers and were removed. Similarly,
semantic retrieval was useful as an experiment but did not beat the final
lexical baseline on our guard. We kept the configuration that generalized
better within the available evidence, not the one that sounded most advanced.

## What's next

We prototyped an offline sparse semantic route and intent-conditioned policy as
ablations. They remain disabled in the final baseline because the baseline
performed better on the held-out public guard. Given more time, we would test
these routes against independently authored customer language and unseen
targets before enabling them for a live catalog.

## Limitations and responsible claims

- The `0.955831` result is on the 200-session public development set; it is not
  a claim about the organizer's 800 private sessions.
- The parser and facet vocabulary are hand-built for the supplied
  Clothing/Shoes/Jewelry catalog and may miss unseen synonyms or categories.
- The final baseline is lexical and deterministic. It does not use an LLM or
  pretrained embedding model, and its optional semantic route is not enabled.
- The agent operates on a frozen catalog and has no live inventory, checkout,
  transaction, or production UI capability.

## Feasibility disclosure

| Item | Disclosure |
|---|---|
| Runtime model | None |
| Network required during scoring | No |
| API credentials | None |
| Token usage | 0 |
| Estimated model cost | $0 |
| Catalog initialization | Approximately 9.6 seconds in the final release run; hardware-dependent |
| Per-turn latency | 39.27 ms mean / 84.88 ms p95 in the final release run; hardware-dependent |
| Python | Python 3.11 declared in `environment.yml`; runtime supports Python 3.10+ |

## Team

EntropyShop was built by Aniket Khan and Ronak Kalvani, both first-year PhD
students in Computer Science at the National University of Singapore (NUS).

## Links

- Source code: https://github.com/ronakkalvani/techjam-conversational-search
- Demo video: **PENDING — add the public YouTube URL before submitting on Devpost**

## Three-minute video script

### 0:00–0:20 — Problem

“A customer has one hidden target among 50,000 products, but they describe
requirements rather than naming the item. We have at most ten turns. The
official BM25 baseline scores 0.107.”

Show the README score table.

### 0:20–0:45 — Insight

“Our experiments showed that learning one, two, and four requirements gives
rapidly increasing hit rates. So the core problem is not just ranking—it is
asking the most valuable next question.”

Show the entropy utility formula and architecture diagram.

### 0:45–1:45 — Live official-protocol session

Run from the repository root:

```bash
conda activate entropyshop
python scripts/demo_session.py --sample-id public_0094
```

Narrate:

“This is the real frozen 50,000-product catalog and the official customer
protocol. The first message identifies mid-calf boots and leather. EntropyShop
returns one cautious recommendation and asks for the next most useful
requirement. After the customer reveals the sole and shaft details, the target
moves to rank one on turn two.”

Pause on the final `SUCCESS` line and product title.

### 1:45–2:20 — Architecture

“The agent unions category and lexical retrieval so one parsing mistake cannot
filter the target out. It reranks with exact phrases and product facets, tracks
overrides explicitly, and demotes products already shown. Its question selector
combines information gain with answerability so it does not waste turns asking
high-entropy questions the customer cannot answer.”

Show `docs/ARCHITECTURE.md` or the README diagram.

### 2:20–2:45 — Evidence

“Across all 200 public sessions, Hit Rate is 1.0, MRR is 0.932, and the final
TechnicalScore is 0.955831. A paraphrase stress test scores 0.928. The runtime
uses zero tokens, costs zero dollars, and averages 39.27 milliseconds per turn
in our final release run.”

Show the results table and robustness table.

### 2:45–3:00 — Close

“EntropyShop demonstrates that a shopping copilot does not need to generate
more—it needs to ask better. It is deterministic, transparent, inexpensive,
and ready for offline evaluation.”

Show the project title, team names, and repository URL.

## Recording checklist

- Use a terminal font large enough for the recommendation titles to be read.
- Start from the repository root with the documented Conda environment
  activated.
- Run the demo once before recording so disk caches are warm.
- Keep the GitHub repository public.
- Show the official-data attribution and make no claim about private-set
  performance.
- Keep the video at or below three minutes.
- Upload the video to YouTube with **public visibility** and paste the final
  link above and into Devpost.
