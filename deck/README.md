# Oblique deck · seed v0

One card per YAML file in `cards/`, named after its ID and validated by `oblique.deck.load_deck` (schema in `src/oblique/deck/schema.py`).

| Cards | `origin` | `family` | Author | Use |
|---|---|---|---|---|
| `OB-001` … `OB-030` | `seed` | one of 8, 3–4 cards each | felipe | seed deck (condition M0) |
| `E6-1` … `E6-6` | `paper` | `null` | felipe | the six openings of Wadhwa et al. (2026), paraphrased (condition E6) |
| `CTRL-R-001` … `CTRL-R-030` | `control` | `null` | random | one random common noun each (condition R) |

All cards are generation 0. Opposite pairs (`opposite_of`, symmetric): OB-009/OB-010, OB-017/OB-018, OB-021/OB-022, OB-028/OB-029.

The `OB-*` and `E6-*` texts are the author's, transcribed word for word from the seed deck v0 list.

## Control cards

The control cards are a perturbation with no strategy behind it: one English noun per card.

- **Source:** the noun list shipped with the [`wonderwords`](https://pypi.org/project/wonderwords/) package, version 2.2.0 (`wonderwords/assets/nounlist.txt`, MIT license).
- **Method**, fixed before any result:
  1. keep lowercase, purely alphabetic entries;
  2. drop entries in the same package's profanity list;
  3. keep common words: English Zipf frequency ≥ 4.0 in [`wordfreq`](https://pypi.org/project/wordfreq/) 3.1.1, about 10 or more occurrences per million words (2,755 nouns remain);
  4. sort the pool and draw 30 words without replacement with the `cards` seed (`oblique.seeds`, master seed 20261026). The i-th word drawn becomes `CTRL-R-{i:03d}`.
- **Regenerate** (rewrites the same 30 files):

  ```bash
  uv run --group deck python -m oblique.deck.controls
  ```

  Both packages are pinned in the `deck` dependency group, because a different version of either list could change the pool.

## License

The deck (`deck/cards/`) is released under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license: see [`LICENSE`](LICENSE). The code that loads and generates it is MIT, like the rest of the repository.
