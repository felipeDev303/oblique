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
- Baseline cost gate (R-20261030-02): Contributor holds 16 concurrent generations with no 429s (18 sets/min), temperature 1.5 is accepted and still parses, and a shared prefix with a different tail is not cached by the provider.
- Baseline runner (`python -m oblique.baseline`): C0 and C1 on 20 split A instances (one per domain plus four, from a new `baseline` seed purpose), with ADR-006 validity and retry, Spark judges for quality (both orders), distance (every pair in a set) and grounding, Wonder for quality agreement, and a secondary distance from local `all-mpnet-base-v2` embeddings. Judge calibration without external labels: position consistency, length bias, Spark–Wonder agreement (exact and Cohen's κ) and Spearman between the two distances. First result (R-20261030-03, US$0.92): C1 − C0 CU +0.32 [0.29, 0.35], driven by distance; q is at the ceiling against the gold.
- Quality measured against an anchor idea from the same model (one K = 1 draw per instance at a reserved `sample_idx`, ADR-008), with q against the gold kept as a secondary metric. Each client now has its own concurrency limit, optionally adaptive (halves on a 429), and a 180 s timeout. Validation (R-20261030-04, US$0.04): q leaves the ceiling, C0 0.58 and C1 0.16, but the judge favours longer ideas (r = 0.50) and C1's ideas are shorter; controlling for length, the C1 effect drops from −0.42 to −0.19 [−0.35, 0.00].
- Length control and sequential generation (ADR-009): `ideation-v3` caps the words of each field, `quality-v2` adds that length and detail are not merits in themselves, and the new baseline C1s writes each idea in its own call, conditioned on the earlier ones with a fixed text. A logistic regression gives the win rate at equal length as a sensitivity analysis. A judgment refused by the provider's content filter now counts as missing instead of stopping the run. Validation (R-20261030-05, US$0.18): C1s matches C0 in length (76 words each), q 0.56 against 0.50, C1s − C0 +0.09 [−0.07, 0.25], and keeps most of C1's diversity.

### Oct 31 (Day 6)

- Content filter rule (ADR-006, amendment 6): one identical retry, then a filtered generation is invalid (`content_filtered`, q = 0) and a filtered judgment is a missing value. A run never stops for a filter.
- Map sweep runner (`python -m oblique.map_sweep`, with a `--plan` mode that counts calls, cost and time without calling the API): anchors, S0 = C1s, and one idea per card and instance, with the `inject-v1` frame around each card; quality against the anchor, grounding and distance to S0. A run resumes from the response cache, and a spend ledger keeps its budget across launches. Reports q controlled for length (joint logistic regression at equal length) and raw q, d, m, grounding, and invalid and filtered rates with CIs, plus the ADR-003 bands.
- Judge concurrency limit that climbs by one after 50 straight successes up to a ceiling and halves on a 429.

## License

Code: [MIT](LICENSE). The card deck (`deck/cards/`) is released under [CC BY 4.0](deck/LICENSE).
