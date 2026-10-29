# CLAUDE.md

## Project

Oblique is a deck of "oblique" perturbation cards applied at inference time. Each idea in a K=3 set is generated under a different card, with the goal of making an LLM (Muse Spark) propose more diverse scientific ideas without losing quality. Nothing is trained. Sets are scored with Creative Utility (greedy, γ = 0.9) against gold ideas, with paired bootstrap CIs over instances. A single core serves two uses: the experiment (with judges) and an OpenAI-compatible proxy (no judges, explicit `+oblique` opt-in).

Models: Muse Spark and Muse Wonder via Meta Model API (`https://api.meta.ai/v1`); Muse Spark Open via Baseten (`meta-models/muse-spark-open-1026`).

Built solo for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4 2026, Indie track, theme "Push the Frontier").

## Planning vault

All planning lives in an Obsidian vault at `../oblique-vault` (relative to this repo). The vault is written in Spanish. Key notes: `03 Marco técnico/Arquitectura.md`, `Diseño experimental.md`, `Creative Utility.md`, `Meta Model API.md`, `05 Gestión/Convenciones.md`, `09 Decisiones/ADR-*.md`.

The vault is **read-only**, except for:
- `07 Bitácora/`: one daily note, from `Plantillas/Plantilla Diario.md`.
- `08 Experimentos/`: one note per run.

## Conventions

- Code, commits, README, prompts and cards: **English**.
- Commits: Conventional Commits (`feat(judges): add pairwise quality judge`, `exp: map split A`). Small commits, one per logical step.
- Branches: `main` always runs; work goes on `feat/*` and `exp/*`.
- Python 3.11, `ruff` + `pytest`, Pydantic for idea cards and deck cards. Managed with `uv`.
- Every run is launched from a config in `configs/R-*.yaml`, never with ad-hoc parameters.
- IDs:
  - seed card `OB-NNN`
  - evolved card `OB-G{gen}-NNN`
  - control `CTRL-{type}-NNN`
  - paper opening `E6-N`
  - condition codes `C0 C1 C2 T R E6 M0 ML MG C1-O ML-O`
  - run `R-YYYYMMDD-NN`
  - instance `i-NNNN`
  - decision `ADR-NNN`
- Run outputs in `runs/R-*/`: `config.yaml`, `generations.jsonl`, `judgments.jsonl`, `metrics.json`, `summary.md`. Every JSONL line carries `run_id`, `instance`, `card`, `model`, `prompt_hash`, `seed`, `tokens_in`, `tokens_out`.

## Non-negotiable rules

1. **New work only.** Everything is written from scratch in this repo during the event. Do not copy code from other projects. Each day, log what was built in the README section "Built during the hackathon".
2. **Never commit secrets.** Keys live only in `.env`. Review `git diff` before every commit.
3. **No Eno & Schmidt text.** Do not include the text, or close paraphrases, of any *Oblique Strategies* card (ADR-002). Do not write deck cards: the author writes them.
4. **No paid API calls without asking first.** State the number of calls and the approximate cost before running anything that hits Model API or Baseten. Tests never call the API (use synthetic fixtures).
5. **Split B is sealed.** Do not read or use it until the author says so.
6. **Experimental design decisions are not the assistant's to make.** On any ambiguity, ask; the author records the decision as an ADR.
7. All model calls go through the local response cache (key = hash of model, messages, params, seed).
