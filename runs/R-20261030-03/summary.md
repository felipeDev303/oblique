# R-20261030-03 · Baseline C0 / C1 and judge calibration

Config: `configs/R-20261030-03.yaml`. 20 split A instances (1 per domain + 4, `baseline` seed). Generator `muse-spark-1.3-contributor` (temperature 1.0, default reasoning, no seed). Judges: Spark Contributor (temperature 0, `minimal`) for quality, distance and grounding; Wonder for quality agreement. Embeddings: `all-mpnet-base-v2@e8c3b32`. 800 calls, **US$0.92** (Wonder US$0.80, priced at unofficial reference rates).

The first launch stopped on sustained 429s from the Spark judges at concurrency 16 (short calls, ~300 requests/min), after all generations and all Wonder judgments. The run was resumed at concurrency 4 from the response cache, with no retries.

## Results (mean, 95% bootstrap CI over 20 instances)

| Condition | CU | q (Spark) | d (LLM) | d (embeddings) | Grounding | Invalid |
|---|---|---|---|---|---|---|
| C0 (3 × K=1) | 0.454 [0.427, 0.485] | 0.992 [0.975, 1.000] | 0.156 [0.111, 0.204] | 0.257 [0.235, 0.282] | 0.825 [0.762, 0.883] | 0/60 |
| C1 (K=3 set) | 0.770 [0.747, 0.792] | 0.983 [0.958, 1.000] | 0.674 [0.651, 0.693] | 0.523 [0.497, 0.546] | 0.733 [0.692, 0.779] | 0/60 |
| **C1 − C0** | **+0.316 [0.287, 0.346]** | −0.008 [−0.025, 0.000] | **+0.518 [0.474, 0.561]** | +0.266 [0.237, 0.295] | | |

- Strict label rule: CU unchanged (all 120 ideas have both labels on the list).
- CU with embedding distance: C0 0.527, C1 0.679.
- Wonder mean q per idea: C0 1.000, C1 0.983.
- Sanity check (C1 d > C0 d): passes.

## Calibration

| Check | Result |
|---|---|
| Spearman, LLM d vs embedding d (120 pairs) | ρ = 0.85 |
| Spark vs Wonder, q per idea (120) | exact 97.5%, κ = 0.39 (low κ from the ceiling: nearly every q is 1) |
| Spark vs Wonder, per order (240) | exact 98.8%, κ = 0.975 |
| Position consistency, Spark | 97.5% consistent; A both 2.5%; B both 0% |
| Position consistency, Wonder | 98.3% consistent; A both 0.8%; B both 0.8% |
| Length bias, Spark (length vs q) | Pearson 0.06, Spearman 0.05 |
| Length bias, Wonder | Pearson 0.08, Spearman 0.08 |
| Mean length shown to the quality judge | idea 959 chars; gold 864 chars |

**Ceiling:** the generated idea beats the gold in both orders for almost every idea, with both judges. q does not separate the conditions, so CU here is driven by distance.
