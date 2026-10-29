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

## License

Code: [MIT](LICENSE). The card deck will be released under CC BY 4.0 in `deck/LICENSE`.
