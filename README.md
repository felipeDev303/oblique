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

## License

Code: [MIT](LICENSE). The card deck will be released under CC BY 4.0 in `deck/LICENSE`.
