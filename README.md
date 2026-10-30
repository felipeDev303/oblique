# Oblique

**A deck of "oblique" perturbations applied at inference time so that an LLM proposes more diverse scientific ideas without losing quality.**

> **Status:** work in progress (built during the Meta Global AI Developer Hackathon, Oct 26 – Nov 4 2026).

## Research question

Can a deck of inference-time perturbations recover an LLM's idea diversity without paying for it in quality, and without training anything?

When LLMs are used for research ideation, they tend to collapse onto a few safe directions. Asking for a *set* of ideas increases diversity but lowers quality. Oblique tests a different approach. Each idea in a set is written under a different "card", a short constraint on *how* to approach the source documents. The resulting set is scored with **Creative Utility** (quality × distance, greedy, γ = 0.9), with paired bootstrap confidence intervals over benchmark instances.

Built on Meta models: Muse Spark (generation), Muse Wonder (deck evolution) and Muse Spark Open (judging, transfer and an open-weight proxy backend).

## Repository layout

```text
configs/            experiment configs (every run is launched from one)
deck/cards/         one card per YAML file
src/oblique/        deck, prompts, llm, cache, generate, judges, metrics, evolve, proxy, export
data/               benchmark (not committed)
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

## License

Code: [MIT](LICENSE). The card deck will be released under CC BY 4.0 in `deck/LICENSE`.
