# R-20261030-01 · Smoke test and cost gate

Config: `configs/R-20261030-01.yaml`. One split A instance (`i-0032`), K = 3. Prompts: ideation-v2, quality-v1, distance-v1, grounding-v1.

| Question | Result |
|---|---|
| Models in `/v1/models` | `muse-spark-1.3`, `muse-spark-1.3-contributor` and `muse-wonder-1.0` all present |
| JSON mode (`response_format: json_object`) | Accepted; 5/5 sets parse as `{"ideas": [...]}`, 15/15 ideas valid, no code fences |
| Taxonomy labels | 15/15 `ok` on both axes |
| `seed` honoured | No: same seed and same request gave different text (temperature 1.0) |
| Provider prompt cache | 2,929 of 2,992 input tokens cached on an immediate identical repeat; 0 on the same messages with another seed, and 0 across models |
| Minimum reasoning effort | `minimal` (`none` rejected with 400 by Spark and Wonder) |
| Reasoning tokens | Included in `completion_tokens`. Generation: Contributor 2,039–3,140; Standard 2,022 by default and 1,141 with `low`. Judges at `minimal`: Spark 200–465, Wonder 55–82 |
| Latency per generation | Contributor 35–53 s; Standard 11–13 s |
| Judges | Spark: q = 1.0 (consistent across orders), d = 0.67, grounding 5/5. Wonder: q = 1.0 |
| Cost | 11 calls, US$0.097 at Standard reference prices (Contributor priced as Standard, so this is an upper bound). The real Contributor price has to be read from Billing |
