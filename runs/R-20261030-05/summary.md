# R-20261030-05 · Length control and sequential generation (ADR-009)

Config: `configs/R-20261030-05.yaml`. 20 instances. New anchors and new C0, C1 and C1s ideas with `ideation-v3`, judged against the anchor with `quality-v2` in both orders, plus `grounding-v1`. Spark Contributor. **US$0.175** in 698 logged calls: US$0.100 in the first launch, stopped by a content-filter 400, and US$0.075 in the resume. No 429s. 20/20 anchors valid, 180/180 ideas valid, labels on the list.

Two judge requests were refused by the provider's content filter and count as missing: grounding on i-0520 C1s slot 2, and quality on i-5677 C1s slot 1 with the anchor first (the same texts in the other order passed). So C1s has n = 19 for q.

## Length (words of gap + claim + method)

| | Mean | SD | Min | P25 | Median | P75 | Max | Idea − anchor | Within caps |
|---|---|---|---|---|---|---|---|---|---|
| Anchor | 73.3 | 9.0 | 59 | 66.8 | 71.5 | 80.2 | 93 | | 100% |
| C0 | 75.8 | 8.9 | 50 | 71.0 | 75.0 | 79.0 | 100 | +2.4 | 100% |
| C1 | 62.9 | 8.3 | 51 | 57.0 | 60.5 | 67.5 | 89 | −10.4 | 100% |
| C1s | 75.5 | 9.4 | 53 | 69.0 | 76.0 | 80.0 | 99 | +2.1 | 100% |

## Length vs q

| | r(words, q) | r(idea − anchor, q) | n |
|---|---|---|---|
| All ideas | 0.24 | 0.43 | 179 |
| C0 | 0.23 | 0.52 | 60 |
| C1 | 0.09 | 0.33 | 60 |
| C1s | −0.03 | 0.22 | 59 |

R-20261030-04 for comparison (characters): 0.50 all and 0.46 in C0 on the raw length; 0.56 in C0 on the difference.

## q against the anchor (mean, 95% bootstrap CI over instances)

| Condition | q | Controlled for length (at equal length) | Ties | Wins | Losses | Consistent | A both | B both |
|---|---|---|---|---|---|---|---|---|
| C0 | 0.500 [0.333, 0.667] | 0.463 [0.312, 0.625] | 16.7% | 41.7% | 41.7% | 83.3% | 11.7% | 5.0% |
| C1 | 0.267 [0.150, 0.400] | 0.380 [0.217, 0.561] | 16.7% | 18.3% | 65.0% | 83.3% | 8.3% | 8.3% |
| C1s | 0.561 [0.430, 0.684] | 0.554 [0.412, 0.694] | 23.7% | 45.8% | 30.5% | 76.3% | 11.9% | 11.9% |
| **C1s − C0** | **+0.088 [−0.070, 0.254]** | **+0.091 [−0.092, 0.269]** | | | | | | |
| C1 − C0 | −0.233 [−0.375, −0.100] | −0.083 [−0.280, 0.108] | | | | | | |
| C1s − C1 | +0.316 [0.219, 0.421] | +0.175 [−0.022, 0.342] | | | | | | |
| All ideas | | | 19.0% | | | 81.0% | 10.6% | 8.4% |

Logistic regression on 358 judgments: condition + (idea − anchor in words), log-odds +0.624 [0.330, 1.004] per 10 words. CIs resample instances.

## Side measures

| | C0 | C1 | C1s | C1s − C0 | C1s − C1 |
|---|---|---|---|---|---|
| Grounding | 0.787 [0.729, 0.842] | 0.721 [0.658, 0.783] | 0.717 [0.667, 0.771] | −0.071 [−0.133, −0.008] | −0.004 [−0.079, 0.067] |
| Embedding d | 0.311 [0.283, 0.343] | 0.572 [0.540, 0.604] | 0.514 [0.482, 0.544] | +0.202 [0.161, 0.240] | −0.059 [−0.094, −0.024] |

No CU here: this validation has no distance judge.
