"""One line of `runs/R-*/generations.jsonl` per idea slot.

Each line carries the run conventions (run_id, instance, card, model,
prompt_hash, seed, tokens) plus validity (ADR-006 §1) and label conformity per
axis (ADR-006 §2). Token counts belong to the API call that produced the set,
so they repeat across the K slots of one call: aggregate cost over unique
`call_key`, not over lines.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from oblique.generate.labels import LabelResult, LabelStatus, classify_method, classify_opportunity
from oblique.generate.schema import IdeaCard
from oblique.generate.validate import InvalidReason, ParsedSet
from oblique.llm import Usage


class GenerationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    instance: str
    condition: str
    card: str | None
    slot: int
    model: str
    prompt_hash: str
    seed: int
    attempt: int = 0
    call_key: str
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0

    valid: bool
    invalid_reason: InvalidReason | None = None
    invalid_detail: str | None = None
    idea: IdeaCard | None = None

    # Label conformity per axis; None when the idea is invalid.
    opportunity_label_status: LabelStatus | None = None
    opportunity_label: str | None = None
    opportunity_label_raw: str | None = None
    method_label_status: LabelStatus | None = None
    method_label: str | None = None
    method_label_raw: str | None = None

    @property
    def conforming(self) -> bool:
        """Valid and both labels ok/other: the condition for q > 0 under the strict rule."""
        return (
            self.valid
            and self.opportunity_label_status in (LabelStatus.ok, LabelStatus.other)
            and self.method_label_status in (LabelStatus.ok, LabelStatus.other)
        )


def _label_fields(axis: str, result: LabelResult) -> dict[str, object]:
    return {
        f"{axis}_label_status": result.status,
        f"{axis}_label": result.canonical,
        f"{axis}_label_raw": result.raw,
    }


def records_from_set(
    parsed: ParsedSet,
    *,
    cards: Sequence[str | None],
    run_id: str,
    instance: str,
    condition: str,
    model: str,
    prompt_hash: str,
    seed: int,
    call_key: str,
    usage: Usage,
    attempt: int = 0,
) -> list[GenerationRecord]:
    """One record per slot. `cards[i]` is the card applied to slot i (None: no card)."""
    if len(cards) != parsed.k:
        raise ValueError(f"need one card entry per slot: {len(cards)} != {parsed.k}")
    shared = {
        "run_id": run_id,
        "instance": instance,
        "condition": condition,
        "model": model,
        "prompt_hash": prompt_hash,
        "seed": seed,
        "attempt": attempt,
        "call_key": call_key,
        "tokens_in": usage.tokens_in,
        "tokens_cached": usage.tokens_cached,
        "tokens_out": usage.tokens_out,
        "tokens_reasoning": usage.tokens_reasoning,
    }
    records = []
    for slot in parsed.slots:
        fields: dict[str, object] = {
            **shared,
            "card": cards[slot.index],
            "slot": slot.index,
            "valid": slot.valid,
            "invalid_reason": slot.reason,
            "invalid_detail": slot.detail,
            "idea": slot.idea,
        }
        if slot.idea is not None:
            fields |= _label_fields("opportunity", classify_opportunity(slot.idea.problem_gap))
            fields |= _label_fields("method", classify_method(slot.idea.method))
        records.append(GenerationRecord.model_validate(fields))
    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel], *, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def read_generations(path: str | Path) -> list[GenerationRecord]:
    with Path(path).open(encoding="utf-8") as f:
        return [GenerationRecord.model_validate_json(line) for line in f if line.strip()]
