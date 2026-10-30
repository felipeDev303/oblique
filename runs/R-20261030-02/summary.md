# R-20261030-02 · Cost gate before the baseline

Config: `configs/R-20261030-02.yaml`. Generator `muse-spark-1.3-contributor`, temperature 1.0, default reasoning, JSON mode, no seed. 84 calls, US$0.065 at the Contributor prices from Billing.

| Question | Result |
|---|---|
| Provider cache, same prefix and different tail (K = 3, then K = 1) | **0** of 2,733 input tokens cached. The only hit in the gate was an identical request repeated (2,929 of 2,975) |
| Highest temperature accepted (≤ 1.5) | **1.5**; JSON parsed in 2/2 sets (3/3 valid ideas each) |
| Concurrency 4 → 16 (2 waves each) | **No 429s, no retries, no failures at any level** |
| Latency per set | p50 32–51 s at every level, flat in concurrency; one outlier of 382 s (a 300 s timeout, then a retry) |
| Throughput at 16 | **18.1 sets/min** (17.6 and 18.6 per wave) |
| Decision | Contributor holds 16 → generator and Spark judges in Contributor |

| Level | Wave | OK | 429 | Wall (s) | p50 / p90 / max (s) | Sets/min |
|---|---|---|---|---|---|---|
| 4 | 0 | 4/4 | 0 | 43.9 | 40.1 / 43.9 / 43.9 | 5.5 |
| 4 | 1 | 4/4 | 0 | 54.7 | 51.5 / 54.7 / 54.7 | 4.4 |
| 8 | 0 | 8/8 | 0 | 50.2 | 41.6 / 47.5 / 50.2 | 9.6 |
| 8 | 1 | 8/8 | 0 | 49.2 | 32.5 / 37.7 / 49.2 | 9.7 |
| 12 | 0 | 12/12 | 0 | 58.0 | 39.8 / 56.7 / 58.0 | 12.4 |
| 12 | 1 | 12/12 | 0 | 65.4 | 41.1 / 55.0 / 65.4 | 11.0 |
| 16 | 0 | 16/16 | 0 | 51.7 | 44.9 / 47.9 / 51.7 | 18.6 |
| 16 | 1 | 16/16 | 0 | 54.5 | 43.2 / 52.4 / 54.5 | 17.6 |
