# R-20261030-04 · Anchor as the quality reference (ADR-008)

Config: `configs/R-20261030-04.yaml`. 20 anchors (K = 1, no card, `sample_idx` 500), then the 120 C0 / C1 ideas of R-20261030-03 re-judged against them with quality-v1, both orders. Spark Contributor, temperature 0, `minimal`. 260 calls, **US$0.042**. Generator limit 16; judge limit 8 (adaptive, not triggered): no 429s, no retries, no parse errors, 20/20 valid anchors.

## q against the anchor (mean, 95% bootstrap CI over 20 instances)

| Condition | q | Ties (0.5) | Wins (1) | Losses (0) | Consistent | A both | B both | q against the gold |
|---|---|---|---|---|---|---|---|---|
| C0 | 0.575 [0.442, 0.708] | 18.3% | 48.3% | 33.3% | 81.7% | 11.7% | 6.7% | 0.992 |
| C1 | 0.158 [0.067, 0.267] | 8.3% | 11.7% | 80.0% | 91.7% | 5.0% | 3.3% | 0.983 |
| **C1 − C0** | **−0.417 [−0.533, −0.300]** | | | | | | | −0.008 |
| All ideas | | 13.3% | | | 86.7% | 8.3% | 5.0% | |

A tie happens only when the same position wins both orders, so ties = 1 − consistency by construction. C0 ideas are draws from the same request as the anchor, so q near 0.5 is the expected value: its CI contains it.

## Length confound (`notebooks/R-20261030-04_length.py`, no API calls)

| | C0 | C1 |
|---|---|---|
| Mean chars shown to the judge (anchor: 1,072) | 1,115 | 803 |
| Idea − anchor | +44 | −269 |
| r(idea − anchor, q), within condition | 0.56 | −0.02 |

- Over all ideas: r(length, q) = 0.50, above the 0.3 threshold set for a length-controlled win rate.
- Linear model on ideas, CI by resampling instances: C1 effect −0.417 [−0.533, −0.300] without control, **−0.188 [−0.350, +0.001]** controlling for length; +0.073 [+0.039, +0.111] per 100 extra characters.
- The r of 0.06 measured against the gold in R-20261030-03 was hidden by the ceiling.

## CU with q against the anchor (LLM distances of R-20261030-03)

| C0 | C1 | C1 − C0 |
|---|---|---|
| 0.356 [0.306, 0.400] | 0.150 [0.063, 0.247] | **−0.206 [−0.295, −0.111]** |

With q against the gold, C1 − C0 was +0.316 [0.287, 0.346].
