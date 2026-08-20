# Data Requirements

Response caches, judge logs, ratings, and pairwise logs are ignored by Git. Tests
do not need them; real runs do. Obtain the reference snapshot separately and use
these hashes to check it.

## Default Kindness snapshot

| File under `data/` | Used for | Reference SHA-256 |
|---|---|---|
| `output/valuearena/processed/full8_kindness/askreddit_1000_responses_completed.json` | Direct-rating collection | `823599486ddbd12c148018c9797b2820ce0469ef419ad658ad855e2686c095b4` |
| `output/numerical_rating/kindness_1000_round_robin/ratings.csv` | Whole-rating analyses | `c6e306badaf2654cc5c9e4ed6e89ab41e3aa939fdb26f7affac77a6eabc31399` |
| `output/numerical_rating/kindness_1000_criterion_round_robin/ratings.csv` | Criterion analyses | `cb027bba4be4a44d3f3778a070e5b81a90f5c281df3d9dbee8a898a6aba223f5` |
| `output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl` | Direct/pairwise comparisons | `e7d3e0b63b706903dc1a50ce1255388b9505f4447d43793e12623c06dcf90a2e` |
| `output/valuearena/raw/runs/8_models/kindness/meta.json` | Scenario uncertainty | `1bb2fbc8a5ef98541ec70755b8a56c97977f82299579fad922cb486547287dce` |

## Supplying data

Commands use the paths above by default. Files may be copied or symlinked.

For collection, copy `numerical_rating/configs/kindness_1000_round_robin.yaml`,
change `source`, and pass the new file with `--config`.

The comparison commands accept external paths directly:

```bash
.venv/bin/python -m numerical_rating.analysis.experiments.direct_vs_pairwise \
  --ratings /data/whole/ratings.csv \
  --criterion-ratings /data/criteria/ratings.csv \
  --evaluations /data/pairwise/evaluations.jsonl

.venv/bin/python -m \
  numerical_rating.analysis.experiments.criterion_direct_vs_btd \
  --ratings /data/criteria/ratings.csv \
  --evaluations /data/pairwise/evaluations.jsonl
```

Collection writes resumable logs under:

```text
runs/numerical_rating/kindness_1000_round_robin/
runs/numerical_rating/kindness_1000_criterion_round_robin/
```

Report commands turn those logs into ignored `ratings.csv` and summary files
under `data/output/numerical_rating/`.
