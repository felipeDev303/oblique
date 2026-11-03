# R-20261103-01 · Confirmation on split B

50 instances, US$3.34, one run.

## H1

**Verdict: inconclusive** (ADR-003).

- CU(ML) - CU(C1s): 0.020 [-0.053, 0.094]; BCa check [-0.052, 0.094].
- Controlled q, ML - C1s: 0.006 [-0.091, 0.100]; BCa check [-0.089, 0.103]; margin -0.05.

## Grounding guard

Grounding, ML - C1s: -0.105 [-0.150, -0.060]; threshold -0.1. Reported; it does not decide H1.

## Exploratory (post-map hypotheses are marked in ADR-010)

- H3, CU(ML) - CU(Ts): 0.084 [0.013, 0.153] (n = 48)
- H3, CU(ML) - CU(R): 0.053 [-0.025, 0.130] (n = 49)
- H4, CU(MG) - CU(ML): 0.014 [-0.054, 0.084] (n = 49)
- H6, CU(E6) - CU(C1s): 0.037 [-0.033, 0.105] (n = 49)
- H6, controlled q, E6 - C1s: 0.033 [-0.056, 0.119]
- H7, CU(E6) - CU(ML): 0.017 [-0.048, 0.085] (n = 49)

## Sensitivity (does not decide H1)

- Raw q, ML - C1s: 0.020 [-0.075, 0.116]
- CU under the strict rule, ML - C1s: 0.020 [-0.052, 0.095]
- CU with embedding distance, ML - C1s: 0.017 [-0.049, 0.086]
- Wonder on 207 ideas: agreement with Spark 66.7%, kappa 0.49 per idea, 78.3%, kappa 0.56 per order.
- q, ML - C1s, on those ideas: Wonder -0.010 [-0.210, 0.190]; Spark 0.040 [-0.140, 0.220].
- H3 with the provider error as q = 0 (ADR-006, amendment 7): CU(ML) - CU(Ts) 0.099 [0.023, 0.173] (n = 49).

## Conditions

| Condition | Sets | CU | q controlled | q raw | q gold | d | Grounding | Invalid | Filtered | Provider error |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 49 | 0.164 [0.108, 0.222] | 0.205 [0.134, 0.292] | 0.163 [0.105, 0.225] | 0.881 [0.813, 0.939] | 0.641 [0.618, 0.663] | 0.752 [0.721, 0.784] | 0 | 0 | 0 |
| C1s | 49 | 0.404 [0.331, 0.478] | 0.468 [0.382, 0.551] | 0.463 [0.371, 0.554] | 0.932 [0.874, 0.976] | 0.641 [0.622, 0.661] | 0.776 [0.740, 0.808] | 0 | 0 | 0 |
| Ts | 48 | 0.332 [0.268, 0.391] | 0.344 [0.266, 0.419] | 0.361 [0.285, 0.434] | 0.833 [0.767, 0.892] | 0.639 [0.614, 0.663] | 0.760 [0.720, 0.800] | 11 | 0 | 1 |
| R | 49 | 0.371 [0.306, 0.438] | 0.401 [0.316, 0.473] | 0.412 [0.333, 0.493] | 0.925 [0.860, 0.976] | 0.640 [0.618, 0.660] | 0.709 [0.677, 0.741] | 0 | 0 | 0 |
| E6 | 49 | 0.441 [0.372, 0.509] | 0.501 [0.408, 0.589] | 0.510 [0.422, 0.602] | 0.922 [0.864, 0.973] | 0.642 [0.620, 0.661] | 0.696 [0.662, 0.730] | 0 | 0 | 0 |
| ML | 49 | 0.424 [0.350, 0.495] | 0.474 [0.377, 0.565] | 0.483 [0.388, 0.578] | 0.932 [0.867, 0.980] | 0.659 [0.642, 0.676] | 0.670 [0.634, 0.706] | 0 | 0 | 0 |
| MG | 49 | 0.438 [0.377, 0.489] | 0.487 [0.412, 0.557] | 0.486 [0.408, 0.558] | 0.922 [0.854, 0.976] | 0.637 [0.620, 0.653] | 0.718 [0.687, 0.748] | 0 | 0 | 0 |

## Post-B exploratory (ADR-011)

Written after this run was read, from its stored records, with no model call. 2,000 resamples. None of it was pre-registered.

- Controlled q, MG - C1s: 0.019 [-0.062, 0.104]

C1s - C1 (sequential set against one call), paired:

- CU: 0.240 [0.170, 0.311] (n = 49)
- Controlled q: 0.262 [0.158, 0.352]
- q against the gold: 0.051 [0.010, 0.099] (n = 49)
- d: -0.000 [-0.023, 0.023] (n = 49); with embeddings -0.035 [-0.057, -0.013]

d within the set, against C1s:

| Condition | Δd | Δd embeddings |
|---|---|---|
| R | -0.001 [-0.024, 0.019] (n = 49) | 0.010 [-0.012, 0.030] |
| E6 | 0.001 [-0.022, 0.021] (n = 49) | 0.017 [-0.004, 0.037] |
| ML | 0.018 [-0.002, 0.040] (n = 49) | 0.009 [-0.017, 0.032] |
| MG | -0.005 [-0.024, 0.013] (n = 49) | 0.005 [-0.016, 0.026] |

Product deck (ADR-010, amendment 3; guard on the mean, ADR-011):

| Deck | ΔCU vs C1s | Controlled Δq | Lower bound | Non-inferior | Mean Δgrounding | Guard |
|---|---|---|---|---|---|---|
| ML | 0.020 | 0.006 | -0.090 | no | -0.105 | no |
| MG | 0.034 | 0.019 | -0.062 | no | -0.058 | yes |
| E6 | 0.037 | 0.033 | -0.056 | no | -0.080 | yes |

**Deck: E6**, marked experimental: no deck passed both checks, so the rule takes the highest controlled Δq.

All three decks use the same 2,000 resamples here, so the lower bound for ML differs a little from the one in H1, which uses 10,000.
