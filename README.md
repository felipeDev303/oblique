# Oblique

**A deck of "oblique" perturbations applied at inference time so that an LLM proposes more diverse scientific ideas without losing quality.**

> **Status:** work in progress (built during the Meta Global AI Developer Hackathon, Oct 26 – Nov 4 2026).

## Research question

Can a deck of inference-time perturbations recover an LLM's idea diversity without paying for it in quality, and without training anything?

When LLMs are used for research ideation, they tend to collapse onto a few safe directions. Asking for a *set* of ideas increases diversity but lowers quality. Oblique tests a different approach. Each idea in a set is written under a different "card", a short constraint on *how* to approach the source documents. The resulting set is scored with **Creative Utility** (quality × distance, greedy, γ = 0.9), with paired bootstrap confidence intervals over benchmark instances.

Built on Meta models: Muse Spark (generation and judging) and Muse Wonder (deck evolution and a second quality judge for agreement checks).

## Data

Instances come from **GiantsBench (test set)**, [`giants2026/GiantsBench-test`](https://huggingface.co/datasets/giants2026/GiantsBench-test) on Hugging Face, introduced in *GIANTS: Generative Insight Anticipation from Scientific Literature*. It is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license, as stated in its dataset card. Each instance has summaries of two parent papers (the corpus) and the insight of a downstream paper that combines them (the gold idea).

- Pinned to revision `6095cb1` and checked by SHA256. The data is not redistributed here: `python -m oblique.bench fetch` downloads it into `data/`.
- Changes: the gold idea is the text inside the `<insight>` block of the `completion` field; the dataset's own `query` prompt is not used (our prompts are in `src/oblique/prompts/`).
- Splits: the 100 most recent instances by arXiv ID (YYMM, then number; the dataset has no date field), with equal allocation over the 16 domains, divided into split A (selection) and split B (confirmation) of 50 each, stratified by domain. The lists are in [`splits/`](splits/). Split B is sealed until the confirmation run.

The project application mentioned "the paper's open benchmark" (Wadhwa et al., 2026). That benchmark has not been released, so Oblique uses GiantsBench-test instead. Absolute numbers are therefore not comparable with the paper's; differences in Creative Utility between conditions on the same instances remain valid.

## Results

All numbers are paired differences over benchmark instances with 95% bootstrap CIs. q is the win rate against an anchor idea from the same model, controlled for length unless it says raw; d is the mean distance within a set; CU is Creative Utility. The full tables are in [`runs/R-20261103-01/summary.md`](runs/R-20261103-01/summary.md).

**H1 is inconclusive** (split B, R-20261103-01, the pre-registered test). The ML deck against the sequential set without cards (C1s): CU +0.020 [−0.053, 0.094], n = 49, and controlled q +0.006 [−0.091, 0.100]. The CU interval crosses 0 and the lower bound of q falls past the −0.05 margin, so the deck is neither shown to help nor to hurt. Within the set, ML's ideas are not clearly farther apart than those of C1s: Δd +0.018 [−0.002, 0.040].

**The grounding guard does not pass.** Grounding of ML against C1s is −0.105 [−0.150, −0.060], below the −0.10 threshold. Every deck loses some grounding against C1s (E6 −0.080, MG −0.058, the random-noun controls −0.066). It is reported and does not decide H1.

**Writing the ideas one at a time is what helps** (C1s − C1, exploratory, ADR-011). The sequential set without cards against the same three ideas asked in one call: CU +0.240 [0.170, 0.311], n = 49, controlled q +0.262 [0.158, 0.352], q against the gold +0.051 [0.010, 0.099], and d −0.000 [−0.023, 0.023]. Quality rises, with the same diversity by the LLM distance judge and slightly lower diversity by embeddings (−0.035 [−0.057, −0.013]). It replicates what R-20261030-05 saw on split A, but it was not a pre-registered hypothesis. The product therefore writes sequentially by default. By the rule fixed before split B, no deck showed quality within the margin, so the product deck is E6, the one with the highest controlled Δq (+0.033 [−0.056, 0.119]), marked experimental.

**Map of the deck** (split A, R-20261031-01). 48 conditions, one idea per card and instance. The rule picks 18 of the 30 seed cards as ML, and no card is confirmed liminal, because every CI of Δq is wider than the margin. 6 of the 10 random-noun controls land in the liminal band, the same share as the seed cards. On split B, ML against the random-noun deck is +0.053 [−0.025, 0.130] in CU, and no deck changes d within the set against C1s.

**Evolution, split A against split B.** On split A, Muse Wonder proposed 15 cards from the map; 13 passed the author's review, 9 of them liminal (MG). Per card against C2 on A, MG averaged +0.051 in controlled Δq and ML −0.086 (simple means, no CI). On split B, as sets of three, MG does not separate from ML: CU +0.014 [−0.054, 0.084], and MG − C1s controlled q +0.019 [−0.062, 0.104]. MG loses less grounding than ML (−0.058 against −0.105). Wonder saw the split A results, so the gap on A was expected to shrink.

**Limitations.**

- One benchmark (GiantsBench-test, not the paper's) and 50 instances per split. The CIs of H1 are about ±0.07 in CU and ±0.10 in q.
- The quality judge is Muse Spark, the same family as the generator, against an anchor from the same model. Muse Wonder agrees with it moderately (κ 0.49 per idea on 207 ideas). There is no human evaluation.
- Length is controlled only by a regression, and longer ideas win more often (+0.21 log-odds per 10 words on split B).
- The content filter removed 13 judgments of one instance, and one generation failed with a persistent provider error (ADR-006, amendments 6 and 7), so most comparisons have n = 49.
- C1s − C1, MG − C1s and the product deck come from analyses written after split B was read (ADR-011). They use only the stored records, and the script first checks that it reproduces the run's `metrics.json` exactly (every set and the controlled q of E6 − C1s); it stops if not.

## Repository layout

```text
configs/            experiment configs (every run is launched from one)
deck/cards/         one card per YAML file
src/oblique/        deck, prompts, llm, cache, generate, judges, metrics, evolve, proxy, export
data/               benchmark (not committed)
splits/             instance IDs of split A and split B
runs/               per-run outputs (only summaries are committed)
web/                visualization
notebooks/          analysis
tests/
```

## OpenAI-compatible proxy

```bash
uv run --group product python -m oblique.proxy.app configs/proxy.yaml
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="any-local-name")
answer = client.chat.completions.create(
    model="muse-spark-1.3+oblique",
    messages=[{"role": "user", "content": "Propose a research idea on ..."}],
)
print(answer.choices[0].message.content)
print(answer.model_extra["oblique"]["cards_used"])
```

- `muse-spark-1.3+oblique` writes three answers in sequence, each under a different card, and returns them in one choice. `oblique.cards_used` lists the cards. `muse-spark-1.3` passes the request through. Any other model is refused.
- **It listens only on localhost, and refuses to start on any other address.** It holds the provider's key and accepts any client key, so anyone who could reach it could spend the budget. Do not put it behind a tunnel or a public port.
- No streaming. It uses Standard pricing, which does not train on your content; Contributor is refused.
- It shares one ledger and one budget with the "bring your papers" app (`configs/product.yaml`). The deck is configurable in `configs/proxy.yaml`, and the default is the app's (E6, experimental).
- Identical requests can return the cached answer at no cost. The cache keeps responses, not requests.

## Built during the hackathon

<!-- Updated daily. Everything in this repository was written during the event. -->

### Oct 28 (Day 3)

- Repository scaffold following the planned architecture: `src/oblique/` packages (deck, prompts, llm, cache, generate, judges, metrics, evolve, proxy, export), `configs/`, `deck/cards/`, `data/`, `runs/`, `web/`, `notebooks/`, `tests/`.
- Project tooling: `pyproject.toml` with uv (Python 3.11, locked dependencies), ruff and pytest configuration.
- `.gitignore` that keeps secrets, benchmark data and raw run outputs out of the repo (only run summaries are committed); `.env.example` for Meta Model API and Baseten keys.
- MIT license for the code.

### Oct 29 (Day 4)

- `cache`: SQLite (WAL) response cache keyed by a hash of model, messages, parameters, sample index and attempt. Only successful responses are stored and entries are never overwritten.
- `llm`: a single OpenAI-compatible client for Meta Model API and Baseten (Muse Spark Open). It retries with exponential backoff and jitter, bounds concurrency with a semaphore, deduplicates identical in-flight requests, records input/cached/output/reasoning tokens, and sends every call through the cache.
- `metrics`: greedy Creative Utility (γ = 0.9), tested against a hand-computed set and the paper's figure 1, plus percentile paired bootstrap CIs over instances (10,000 resamples). All seeds derive from one master seed (20261026).
- Schemas: `IdeaCard` (problem_gap, core_claim, method, inspiration) and deck `Card` with a closed family list, symmetric `opposite_of`, and a deck loader that validates every YAML file and the deck as a whole.

### Oct 30 (Day 5)

- Idea validity and taxonomy labels (ADR-006): `parse_set()` separates structural validity (decides q = 0) from label conformity. Labels are normalized and classified as ok / other / off_list / missing per axis.
- `generations.jsonl` records with per-axis label status, and a strict-rule option for Creative Utility used as a pre-registered sensitivity analysis.
- Benchmark loader for GiantsBench-test (CC BY 4.0), pinned by revision and SHA256, with gold insight extraction. Reproducible A/B splits of 50 instances (most recent per domain, stratified, drawn from the `splits` seed); split B is sealed behind an explicit environment flag.
- Versioned prompts written for this project from a specification (no text taken from the paper): `ideation-v2` (always "Propose {k} distinct research ideas", four fields, taxonomy labels at the start of `problem_gap` and `method`, each label defined in one line written for this project, JSON-only `{"ideas": [...]}`), `quality-v1` (pairwise against the gold idea, both orders, idea shown as Problem / Claim / Approach without labels or inspiration), `distance-v1` (1–5 similarity per field) and `grounding-v1` (1–5 dependence of the claim on its sources). Each prompt's hash is pinned in the tests, so an edited template needs a new version. They follow the structure of Wadhwa et al. (2026) and the taxonomy of Chen et al. (2026).
- The response cache key now includes the provider, so the same model and request served by different endpoints never share an entry.
- Smoke test and cost gate (`python -m oblique.smoke`, run R-20261030-01): checks every configured model against `/v1/models` before any paid call, stops new calls past a spending cap, and measures seed behaviour, provider prompt caching, JSON mode, Contributor vs Standard, the minimum reasoning effort and reasoning tokens per call. First real calls to Muse Spark and Muse Wonder: 11 calls, about US$0.10 at Standard reference prices.
- Seed deck v0 in `deck/cards/` (CC BY 4.0): the author's 30 seed cards (8 families, 4 opposite pairs) and 6 paper-opening cards, transcribed word for word, plus 30 random-noun control cards drawn reproducibly from a public noun list (method in `deck/README.md`). A YAML card writer that keeps the text exact.
- Baseline cost gate (R-20261030-02): Contributor holds 16 concurrent generations with no 429s (18 sets/min), and temperature 1.5 is accepted and still parses.
- Baseline runner (`python -m oblique.baseline`): C0 and C1 on 20 split A instances (one per domain plus four, from a new `baseline` seed purpose), with ADR-006 validity and retry, Spark judges for quality (both orders), distance (every pair in a set) and grounding, Wonder for quality agreement, and a secondary distance from local `all-mpnet-base-v2` embeddings. Judge calibration without external labels: position consistency, length bias, Spark–Wonder agreement (exact and Cohen's κ) and Spearman between the two distances. First result (R-20261030-03, US$0.92): C1 − C0 CU +0.32 [0.29, 0.35], driven by distance; q is at the ceiling against the gold.
- Quality measured against an anchor idea from the same model (one K = 1 draw per instance at a reserved `sample_idx`, ADR-008), with q against the gold kept as a secondary metric. Each client now has its own concurrency limit, optionally adaptive (halves on a 429), and a 180 s timeout. Validation (R-20261030-04, US$0.04): q leaves the ceiling, C0 0.58 and C1 0.16, but the judge favours longer ideas (r = 0.50) and C1's ideas are shorter; controlling for length, the C1 effect drops from −0.42 to −0.19 [−0.35, 0.00].
- Length control and sequential generation (ADR-009): `ideation-v3` caps the words of each field, `quality-v2` adds that length and detail are not merits in themselves, and the new baseline C1s writes each idea in its own call, conditioned on the earlier ones with a fixed text. A logistic regression gives the win rate at equal length as a sensitivity analysis. A judgment refused by the provider's content filter now counts as missing instead of stopping the run. Validation (R-20261030-05, US$0.18): C1s matches C0 in length (76 words each), q 0.56 against 0.50, C1s − C0 +0.09 [−0.07, 0.25], and keeps most of C1's diversity.

### Oct 31 (Day 6)

- Content filter rule (ADR-006, amendment 6): one identical retry, then a filtered generation is invalid (`content_filtered`, q = 0) and a filtered judgment is a missing value. A run never stops for a filter.
- Map sweep runner (`python -m oblique.map_sweep`, with a `--plan` mode that counts calls, cost and time without calling the API): anchors, S0 = C1s, and one idea per card and instance, with the `inject-v1` frame around each card; quality against the anchor, grounding and distance to S0. A run resumes from the response cache, and a spend ledger keeps its budget across launches. Reports q controlled for length (joint logistic regression at equal length) and raw q, d, m, grounding, and invalid and filtered rates with CIs, plus the ADR-003 bands.
- Judge concurrency limit that climbs by one after 50 straight successes up to a ceiling and halves on a 429.

### Nov 1 (Day 7)

- Map of the deck on split A (R-20261031-01, US$3.15, 16,895 calls): 48 conditions over 50 instances, classified with the bands fixed in ADR-003. The selection rule gives an ML of 18 of the 30 seed cards, and no liminal card reaches the confirmed sublabel, because the CI of Δq is wider than the margin. 6 of the 10 random-noun controls also land in the liminal band, the same share as the seed cards. Results in [`runs/R-20261031-01/summary.md`](runs/R-20261031-01/summary.md).
- The map summary now prints CIs for the invalid and filtered rates and lists the confirmed liminal cards.
- Evolution G1 (ADR-010, `python -m oblique.evolve.g1`, run R-20261101-01, US$0.23): one call to Muse Wonder with the map as data, which proposes 15 cards with family, parents and a justification. The call is saved verbatim in [`deck/evolution/G1/`](deck/evolution/G1/). The runner checks the deck rules and reports the closest existing card for review. The cards are not measured yet.
- First slice of "bring your papers" (`python -m oblique.product.app configs/product.yaml`): upload 2–5 PDFs and get a sequential K = 3 set from Muse Spark Standard, with each idea's card and sources visible. Each cited quote is checked against the text of the PDF. Limits per session and an overall budget in a persistent ledger protect the credits. The PDFs go inline, stay in memory only for the set, and never enter the local cache. The deck is provisional until the confirmation run (ADR-010).

### Nov 2 (Day 8)

- The author's ADR-002 review discarded 2 of the 15 G1 cards; the other 13 entered the deck. Measured on split A (`python -m oblique.evolve.measure`, R-20261102-01, US$0.97) with the anchors, S0 and C2 of the map, in a joint regression with the map's stored judgments. 9 cards are liminal and form MG, 1 of them confirmed, 4 are inert and none is destructive. Wonder saw this split's results, so the effect may shrink on split B. Results in [`runs/R-20261102-01/summary.md`](runs/R-20261102-01/summary.md).
- "Bring your papers" now reads each PDF in its own call, one at a time, into a document sheet (title, summary, methods, limitations, quoted evidence) and writes the ideas over the sheets. Sources cite evidence by ID, and each quote is checked against the text of its PDF. A PDF past 50 pages is refused whole, with no call. Real trial (R-20261102-02, US$0.27, CC BY papers from arXiv, IDs in the [summary](runs/R-20261102-02/summary.md)): a 94-page PDF was refused at no cost, a first set cost US$0.22 and a second set over the same PDFs US$0.05.
- OpenAI-compatible proxy (`python -m oblique.proxy.app configs/proxy.yaml`, real trial R-20261102-03, US$0.02): `POST /v1/chat/completions` with `model: "muse-spark-1.3+oblique"` writes three answers in sequence under different cards and returns them in one choice, with `oblique.cards_used`. `muse-spark-1.3` passes the request through. No streaming, Standard only, with a configurable deck, and the same ledger and budget as the app.
- Runner for the confirmation on split B (`python -m oblique.confirm`, config R-20261103-01), with a `--plan` gate that reads only the instance IDs, so split B stays sealed. The card of every idea is drawn with the `cards` seed and committed before B is opened. A finished run refuses to start again, and a cut run resumes without asking twice for anything that finished. A BCa check joins the percentile CIs for H1.
- Confirmation on split B (R-20261103-01, US$3.34, 7,617 calls, 7 conditions over 50 new instances). H1 is inconclusive: CU(ML) − CU(C1s) is +0.020 [−0.053, 0.094] and controlled q is +0.006 [−0.091, 0.100], so the lower bound falls past the −0.05 margin. ML loses grounding against C1s (−0.105), past the guard. One generation failed with a persistent provider error (ADR-006, amendment 7), and the content filter blocked 13 judgments of one instance, which leaves the paired comparisons with n = 49. Results in [`runs/R-20261103-01/summary.md`](runs/R-20261103-01/summary.md).
- Analyses after split B (ADR-011, `python -m oblique.post_b`): from the stored records of the run, with no model call. The rebuilt state must reproduce the run's own sets and regression before anything new is computed. They add controlled q for MG − C1s, C1s − C1, d within the set against C1s and the product deck rule. The deck that comes out is E6, marked experimental.
- "Bring your papers" writes sequential sets by default with the E6 deck, marked experimental on the page, and the cards can be turned off. A side-by-side view puts one call for three ideas with no card (K=3) next to the Oblique sequential set.

## License

Code: [MIT](LICENSE). The card deck (`deck/cards/`) is released under [CC BY 4.0](deck/LICENSE).
