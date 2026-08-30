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

EntropyShop reframes conversational product search as active preference
elicitation. Instead of spending every turn generating more prose, it maintains
a belief over 50,000 real products, ranks its best candidates, and asks the
clarification question expected to remove the most uncertainty. It runs fully
offline, uses no LLM at runtime, reports zero tokens, and achieves a 0.955831
TechnicalScore on the 200-session public development benchmark.

## Inspiration

The baseline treats each customer message as another search query. Our early
experiments showed that this misses the most valuable part of the interaction:
the agent controls which preference it asks the customer to reveal next.
Knowing one, two, and four requirements produced sharply improving hit rates,
so we designed the system around acquiring useful evidence quickly rather than
making ranking alone increasingly complicated.

The lexical-questioning concept was proposed by Aniket Khan, inspired by
Akinator's question-driven narrowing of a hidden answer. EntropyShop adapts
that idea to structured product attributes and expected information gain.

## What it does

For every customer turn, EntropyShop:

1. parses new constraints and intent overrides into isolated session state;
2. retrieves candidates through both category and lexical routes;
3. ranks them with phrase containment, facet coverage, conflict penalties, and
   deterministic priors;
4. returns its current best recommendations; and
5. chooses the next clarification field using information gain tempered by
   answerability, expected ranking improvement, missing-value probability, and
   turn cost.

The customer receives recommendations and a useful question in the same turn.
Previously rejected products are demoted, while intent overrides clear stale
suppression and reactivate exploration.

## How we built it

- Python 3.11 and the standard library for the runtime
- Immutable in-memory indexes over the frozen 50,000-product catalog
- Turn-aware constraint parsing and explicit override semantics
- Category-bucket plus IDF-weighted lexical retrieval
- Two-stage deterministic ranking
- Entropy and expected-residual-entropy question selection
- Official evaluator integration, ablation scripts, robustness tests, and 57
  automated tests

The runtime has no external service, API key, model download, or network
dependency.

## Devpost disclosure fields

- **Track:** Track 4 — Shopping Copilot: AI Conversational Search and
  Recommendations
- **Development tools:** Python 3.11, Conda, Git, and pytest
- **APIs:** None
- **Libraries:** Python standard library at runtime; pytest for development
  tests only
- **Dataset and assets:** The organizer's frozen 50,000-product catalog and
  200 public development sessions from the participant kit, derived from
  Amazon Reviews 2023. No images, videos, user identifiers, reviews, private
  sessions, or external model assets are used by the runtime.
- **Runtime configuration:** Baseline mode with intent-conditioned and
  semantic ablation flags disabled; no network and zero model tokens.

## AI assistance disclosure

ChatGPT was used during development as an engineering assistant for
brainstorming, code drafting and review, debugging, test design, evaluation
analysis, and documentation editing. The team made and reviewed the final
design and implementation decisions. No LLM is called by the submitted runtime;
the agent is deterministic, offline, and reports zero model tokens. See
`docs/AI_ASSISTANCE.md` for the full disclosure.

## Challenges

The hardest issue was not retrieval—it was deciding which question was worth a
turn. Raw entropy prefers attributes such as brand because they partition the
catalog well, even when the simulated customer cannot answer them. We added an
answerability-aware customer model so information gain is rewarded only when
the answer is likely to reveal usable evidence.

Another challenge was avoiding overfitting to the public simulator's wording.
A paraphrase stress test exposed that an opening such as “Hoping to find…” was
not parsed as a category. We fixed the structure rather than accumulating more
templates: parsing is now turn-aware, treating the first message as an opening
and subsequent non-refusal messages as evidence.

## Accomplishments

- Hit Rate@10: 1.000 on all 200 public sessions
- MRR: 0.932103
- MTTC: 2.19 turns
- TechnicalScore: 0.955831 versus the 0.10671 official baseline
- 0 reported tokens and $0 model cost
- Fully deterministic and offline
- 0.92838 TechnicalScore under the paraphrased-customer stress test
- 57 passing automated tests, including split safeguards and the official
  evaluator tests
- 1.000 Hit Rate on every deterministic 40-session fold, with TechnicalScore
  ranging from 0.944321 to 0.966583

## What we learned

In multi-turn recommendation, acquiring the right evidence can be more valuable
than adding a more expensive model. We also learned that seemingly reasonable
signals must be measured: the aggregate profile prior reduced public-set
performance, while a ramped recommendation budget substantially improved MRR.
Both findings changed the shipped configuration.

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

- Ronak Kalvani — first-year PhD student in Computer Science at the National
  University of Singapore (NUS); implementation, retrieval, ranking,
  adaptive questioning, parsing, evaluation, testing, and technical
  documentation.
- Aniket Khan — first-year PhD student in Computer Science at the National
  University of Singapore (NUS); originated the lexical-questioning idea
  inspired by Akinator, problem framing, experiment design, solution review,
  and demo and presentation support.

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
