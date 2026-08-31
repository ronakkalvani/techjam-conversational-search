# EntropyShop - Three-Minute Video Production Plan

## Creative direction

- Format: 16:9, 1920x1080, 30 fps, final duration 2:55-3:00.
- Story: hidden-target problem -> information-acquisition insight -> real protocol demo -> auditable architecture -> evidence -> robustness and feasibility -> memorable close.
- Visual style: dark navy background, off-white type, electric cyan for information flow, amber for evidence, violet for uncertainty. Use clean diagrams and terminal footage rather than stock photography.
- Typography: Space Grotesk or Sora for headings; Inter for body and data labels; a large monospace font for terminal footage.
- Trademark safety: use original EntropyShop graphics and text. Do not use Amazon or TikTok logos or unrelated copyrighted footage.
- Claim discipline: label all benchmark results as results on the 200-session public development set; never imply they predict the organizer's 800 private sessions.

## Final scene-by-scene plan

### Scene 1 - The constraint (0:00-0:18, 18 seconds)

Narration:

> Shopping requests rarely name the product. Here, a customer has one hidden target among 50,000 catalog items, reveals requirements over time, and gives us at most ten turns. The official BM25 baseline scores just 0.107.

Visuals:

- Start with 50,000 small candidate dots converging toward one highlighted target.
- Introduce a ten-segment turn counter beside the candidate field.
- End on a compact baseline score card.

On-screen text:

- `1 hidden target / 50,000 products / <=10 turns`
- `Official BM25 baseline: 0.10671 TechnicalScore`

Recording needed: none; use a Canva motion slide.

Judge purpose: establishes the problem, scale, and scoring pressure immediately.

### Scene 2 - The insight (0:18-0:40, 22 seconds)

Narration:

> Our headroom study exposed the real lever: with one, two, and four known requirements, Hit Rate rises from 0.325 to 0.675 to 0.935 - even with a crude ranker. So EntropyShop treats shopping as active preference elicitation: maintain uncertainty over candidates, then ask the question with the highest expected value.

Visuals:

- Animate a three-bar chart: one, two, and four known requirements.
- Transform the bars into a simple posterior funnel.
- Show a compact information-gain expression, then emphasize `expected value`, not raw entropy alone.

On-screen text:

- `Known requirements -> Hit Rate@10`
- `1 -> 0.325 | 2 -> 0.675 | 4 -> 0.935`
- `Ask the question that improves the next ranking most`

Recording needed: none; use a Canva chart and a simple candidate animation.

Judge purpose: carries Innovation & Problem Insight with a measured reason for the design.

### Scene 3 - Official-protocol demo (0:40-1:30, 50 seconds)

Narration:

> This recording uses the frozen 50,000-product catalog and the official customer protocol. The shopper asks for mid-calf boots and gives one clue: leather. EntropyShop resolves the category, ranks candidates, returns one cautious recommendation, and asks for another requirement in the same response. The customer then reveals a synthetic sole and an eight-inch shaft. On turn two, the intended Ariat boot moves to rank one, and the official stopping rule records success. Nothing here is scripted around the target: the same Agent interface, state, retrieval, ranking, and evaluator protocol run across every session.

Visuals:

- Full-screen terminal recording.
- Use subtle Canva overlays to highlight `50,000 real products`, the turn-1 question, the turn-2 rank-1 result, and the final `SUCCESS` line.
- Add a small persistent lower third: `Frozen catalog | Official customer protocol | public_0094`.

On-screen text:

- Keep the terminal as the main text.
- Overlay only: `Turn 1: category + leather`, `Turn 2: two more requirements`, `Target -> rank 1`.

Screen recording required:

```bash
conda activate entropyshop
python scripts/demo_session.py --sample-id public_0094
```

Capture instructions:

1. Record from the repository root at 1920x1080.
2. Use a 22-26 px monospace font, dark background, high contrast, and a terminal width near 92 characters.
3. Run once before recording so file-system caches are warm.
4. Record the command, then use a clean jump cut across the roughly ten-second catalog initialization wait.
5. Scroll or zoom so the first customer message, `ASK FIELD > other`, both turn-2 recommendations, and `SUCCESS` are readable.
6. Hold the final target title and rank for two seconds.

Judge purpose: satisfies the end-to-end demonstration requirement with the real Agent and protocol.

### Scene 4 - Why the system works (1:30-2:05, 35 seconds)

Narration:

> Under the hood, a turn-aware parser updates explicit session state, including refusals and intent changes. Category and lexical retrieval are unioned, so one parsing miss cannot remove the target. A two-stage ranker combines phrase evidence, facets, conflict penalties, and deterministic priors. Finally, the question selector scores information gain together with answerability and expected rank improvement. That matters because a brand question can split the catalog beautifully, yet still waste a turn if the customer cannot answer it.

Visuals:

- Animate a five-stage pipeline: message -> parser and state -> category plus lexical union -> deterministic ranker -> recommendations plus next question.
- Branch the final ranker output into both recommendations and the question selector.
- Add a small `brand trap` callout: `high raw entropy + low answerability = wasted turn`.

On-screen text:

- `Category route U Lexical route`
- `Phrase evidence + facets + conflict penalties`
- `Question utility = information gain + answerability + expected rank gain - turn cost`
- `Optional semantic route: measured, disabled in the submitted configuration`

Recording needed: none. Build this as an original Canva diagram; do not show the optional semantic route in the main flow.

Judge purpose: emphasizes deliberate architecture, reliability, and non-obvious technical decisions.

### Scene 5 - Evidence, not decoration (2:05-2:30, 25 seconds)

Narration:

> On all 200 public development sessions, EntropyShop reaches 1.000 Hit Rate at ten, 0.932 MRR, and a 0.955831 TechnicalScore. The ablation trail shows where it came from: category routing, facet and phrase evidence, adaptive questions, exploration, and a staged recommendation budget - not one opaque model.

Visuals:

- Three large metric cards: Hit Rate@10, MRR, TechnicalScore.
- Under them, animate a compact ablation staircase from baseline to final.
- Keep the public-development caveat visible throughout.

On-screen text:

- `200 public development sessions`
- `Hit Rate@10 1.000 | MRR 0.932103 | TechnicalScore 0.955831`
- Ablation labels: `0.107 baseline -> 0.407 category -> 0.821 facets -> 0.862 phrases -> 0.875 questions -> 0.887 exploration -> 0.956 final`
- Fine print: `Public development result; not a claim about the 800 private sessions.`

Recording needed: none; use charts built from the committed experiment tables and `results.json`.

Judge purpose: targets Technical Execution and shows that each major choice was measured.

### Scene 6 - Robust and practical (2:30-2:50, 20 seconds)

Narration:

> We also tested paraphrased customer language. A failure exposed template overfitting; making parsing turn-aware lifted the stress score from 0.742 to 0.928. The submitted runtime is deterministic, offline, uses zero model tokens, and averages 39.27 milliseconds per turn on our recorded machine.

Visuals:

- Before-and-after bars for the paraphrase stress test.
- Four compact proof chips: `57 tests`, `0 tokens`, `no network`, `39.27 ms mean`.
- Optional one-second terminal still showing `57 passed`.

On-screen text:

- `Paraphrase stress: 0.74187 -> 0.92838`
- `Structural fix: turn-aware parsing`
- `57 tests | deterministic | offline | $0 model cost`
- Fine print: `Latency is hardware-dependent.`

Optional screen recording:

```bash
python -m pytest -q
```

Record only the final `57 passed` result or use a still; do not spend video time waiting for the test run.

Judge purpose: addresses robustness and feasibility with a candid failure-and-fix story.

### Scene 7 - Close (2:50-3:00, 10 seconds)

Narration:

> EntropyShop's lesson is simple: a shopping copilot does not need to generate more. It needs to ask better.

Visuals:

- Return to the candidate-dot motif; all uncertainty collapses onto one target.
- Hold the project title, team, and repository URL for the final three seconds.

On-screen text:

- `EntropyShop`
- `Ask better. Find faster.`
- `Ronak Kalvani + Aniket Khan · NUS`
- `github.com/ronakkalvani/techjam-conversational-search`

Recording needed: none.

Judge purpose: leaves one memorable, technically faithful thesis.

## Canva page map

Build a seven-page 16:9 presentation, with each page matching one scene above:

1. Hidden target and ten-turn constraint.
2. Requirement headroom chart and active-elicitation insight.
3. Framing page/video container for the terminal demo.
4. Simplified submitted architecture and `brand trap` callout.
5. Public-development metrics and ablation staircase.
6. Paraphrase robustness and feasibility proof chips.
7. EntropyShop close card.

The deck should remain legible when viewed as a YouTube video on a laptop: no body text below roughly 28 px, no more than one main claim per page, and no dense screenshots except the deliberately enlarged terminal demo.

## Canva build sequence

1. Generate the seven-page presentation from the page map and exact slide copy above. Use a custom dark technical style because no Canva brand kit is connected.
2. Select the candidate with the clearest data hierarchy and least stock imagery, then create the editable Canva design.
3. Replace any generated benchmark values with the exact values in this plan.
4. Upload the terminal recording and place it full-bleed on page 3; crop out the initialization wait and add the three annotation overlays.
5. Add simple, consistent animations: candidate convergence, bar growth, pipeline reveal, and metric count-up. Avoid decorative spins, bounces, or frequent transitions.
6. Record or upload the narration, then set each page duration to the scene timestamps. Keep two to three frames of audio breathing room at every cut.
7. Add low-volume, properly licensed instrumental music only if it does not compete with speech. Duck it under the demo and narration.
8. Export as 1080p MP4 and verify the final runtime, terminal readability, audio peaks, metric labels, public-set caveat, and final URL before uploading to public YouTube.

## Final quality gate

- Runtime is at or below 3:00; target 2:57-2:59 after final audio edits.
- The primary demo is continuous in logic even if the initialization wait is jump-cut.
- Every benchmark number is exact and labeled as public development or stress-test evidence.
- The semantic route is not presented as part of the submitted configuration.
- No claim implies live inventory, checkout, production UI, or private-set performance.
- Voice is calm and confident at roughly 125-130 words per minute.
- Captions are enabled and manually checked for product names, `TechnicalScore`, `MRR`, and `public_0094`.
- The uploaded YouTube video is public and its URL is added to Devpost and `docs/SUBMISSION.md`.
