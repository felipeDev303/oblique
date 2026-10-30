"""Schema tests. Card texts here are placeholders, not deck cards."""

from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from oblique.deck import Card, DeckError, Family, load_deck, validate_deck
from oblique.generate import IdeaCard

TEXT = "placeholder text for tests"


def card(id_: str, **kw) -> Card:
    origin = kw.pop("origin", None)
    if origin is None:
        origin = (
            "evolved"
            if id_.startswith("OB-G")
            else "seed"
            if id_.startswith("OB-")
            else "control"
            if id_.startswith("CTRL")
            else "paper"
        )
    defaults = {
        "text": TEXT,
        "family": Family.error if origin in ("seed", "evolved") else None,
        "generation": int(id_.split("-")[1][1:]) if origin == "evolved" else 0,
        "author": "felipe",
        "created": date(2026, 10, 29),
    }
    return Card(id=id_, origin=origin, **{**defaults, **kw})


# --- IdeaCard ---------------------------------------------------------------

IDEA = {
    "problem_gap": "[Evidence Gap] text",
    "core_claim": "text",
    "method": "[Empirical Mapping] text",
    "inspiration": "text",
}


def test_idea_card_keeps_text_verbatim():
    idea = IdeaCard.model_validate({**IDEA, "core_claim": "  spaced  "})
    assert idea.core_claim == "  spaced  "


def test_idea_card_ignores_extra_fields():
    assert IdeaCard.model_validate({**IDEA, "title": "x"}) == IdeaCard.model_validate(IDEA)


@pytest.mark.parametrize("field", list(IDEA))
def test_idea_card_requires_every_field(field):
    with pytest.raises(ValidationError):
        IdeaCard.model_validate({k: v for k, v in IDEA.items() if k != field})


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", 3, None, ["x"]])
def test_idea_card_rejects_blank_or_non_text(bad):
    with pytest.raises(ValidationError):
        IdeaCard.model_validate({**IDEA, "method": bad})


# --- Card -------------------------------------------------------------------


def test_family_slugs():
    assert [f.value for f in Family] == [
        "subtraction",
        "inversion",
        "scale",
        "role",
        "constraint",
        "ambiguity_specificity",
        "error",
        "time",
    ]


@pytest.mark.parametrize("id_", ["OB-007", "OB-G1-007", "CTRL-R-003", "E6-4"])
def test_valid_ids(id_):
    assert card(id_).id == id_


@pytest.mark.parametrize("id_", ["OB-7", "OB-0007", "OB-G-007", "CTRL-T-001", "E6-7", "ob-007"])
def test_invalid_ids(id_):
    with pytest.raises(ValidationError):
        card(id_, origin="seed")


def test_seed_and_evolved_need_a_family():
    with pytest.raises(ValidationError, match="family required"):
        card("OB-001", family=None)
    with pytest.raises(ValidationError, match="family required"):
        card("OB-G1-001", family=None)


def test_unknown_family_is_rejected():
    with pytest.raises(ValidationError):
        card("OB-001", family="paradox")


def test_control_and_paper_have_no_family():
    assert card("CTRL-R-001").family is None
    assert card("E6-1").family is None
    with pytest.raises(ValidationError, match="must be null"):
        card("CTRL-R-001", family=Family.error)
    with pytest.raises(ValidationError, match="must be null"):
        card("E6-1", family=Family.inversion)


def test_only_evolved_cards_have_parents():
    assert card("OB-G1-001", parents=["OB-001"]).parents == ["OB-001"]
    with pytest.raises(ValidationError, match="parents"):
        card("OB-001", parents=["OB-002"])


def test_id_must_match_origin_and_generation():
    with pytest.raises(ValidationError, match="origin"):
        card("OB-001", origin="evolved", generation=0)
    with pytest.raises(ValidationError, match="generation"):
        card("OB-G2-001", generation=1)
    with pytest.raises(ValidationError, match="generation"):
        card("OB-001", generation=1)


def test_card_cannot_oppose_itself():
    with pytest.raises(ValidationError, match="own opposite"):
        card("OB-001", opposite_of=["OB-001"])


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        card("OB-001", opuesta_a="OB-002")


# --- Deck -------------------------------------------------------------------


def test_symmetric_opposites_pass():
    deck = {
        "OB-007": card("OB-007", opposite_of=["OB-012"]),
        "OB-012": card("OB-012", opposite_of=["OB-007"]),
    }
    validate_deck(deck)


def test_multiple_opposites():
    deck = {
        "OB-001": card("OB-001", opposite_of=["OB-002", "OB-003"]),
        "OB-002": card("OB-002", opposite_of=["OB-001"]),
        "OB-003": card("OB-003", opposite_of=["OB-001"]),
    }
    validate_deck(deck)


def test_asymmetric_opposites_fail():
    deck = {"OB-007": card("OB-007", opposite_of=["OB-012"]), "OB-012": card("OB-012")}
    with pytest.raises(DeckError, match="symmetric"):
        validate_deck(deck)


def test_opposite_must_exist():
    with pytest.raises(DeckError, match="not in deck"):
        validate_deck({"OB-G1-001": card("OB-G1-001", opposite_of=["OB-099"])})


def test_evolved_card_may_oppose_an_existing_card():
    deck = {
        "OB-001": card("OB-001", opposite_of=["OB-G1-001"]),
        "OB-G1-001": card("OB-G1-001", parents=["OB-001"], opposite_of=["OB-001"]),
    }
    validate_deck(deck)


def _write(directory, c: Card, name: str | None = None):
    data = c.model_dump(mode="json")
    (directory / f"{name or c.id}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_deck_reads_yaml_with_unquoted_dates(tmp_path):
    (tmp_path / "OB-007.yaml").write_text(
        "id: OB-007\n"
        f'text: "{TEXT}"\n'
        "family: ambiguity_specificity\n"
        "origin: seed\n"
        "generation: 0\n"
        "parents: []\n"
        "opposite_of: [OB-012]\n"
        "author: felipe\n"
        "created: 2026-10-29\n",
        encoding="utf-8",
    )
    _write(tmp_path, card("OB-012", opposite_of=["OB-007"]))
    deck = load_deck(tmp_path)
    assert sorted(deck) == ["OB-007", "OB-012"]
    assert deck["OB-007"].family is Family.ambiguity_specificity
    assert deck["OB-007"].created == date(2026, 10, 29)


def test_load_deck_reports_every_problem(tmp_path):
    _write(tmp_path, card("OB-001"), name="OB-002")  # wrong file name
    (tmp_path / "OB-003.yaml").write_text("id: OB-003\norigin: seed\n", encoding="utf-8")
    with pytest.raises(DeckError) as exc:
        load_deck(tmp_path)
    assert len(exc.value.problems) == 2


def test_load_deck_checks_symmetry(tmp_path):
    _write(tmp_path, card("OB-001", opposite_of=["OB-002"]))
    _write(tmp_path, card("OB-002"))
    with pytest.raises(DeckError, match="symmetric"):
        load_deck(tmp_path)
