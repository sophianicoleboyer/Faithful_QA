## Curated Benchmark: Evidence-Grounded QA Faithfulness & Robustness

This repository contains a **curated benchmark dataset** (unified schema) and the **scripts** used to inspect and normalize multiple evidence-grounded QA / attribution datasets.

### What’s in this repo
- **Dataset card / report**: `curated_benchmark/dataset_card.md`
- **Stats**: `curated_benchmark/stats.json`
- **Scripts**
  - `curated_benchmark/scripts/inspect_datasets.py`
  - `curated_benchmark/scripts/normalize_datasets.py`
  - `curated_benchmark/scripts/package_release.py`

### Download the benchmark (recommended)
The benchmark file is large, so it is distributed as a compressed artifact:

- `benchmark_eval.jsonl.gz`
- `SHA256SUMS.txt`

Download (PolyBox): `https://polybox.ethz.ch/index.php/s/BSpYDY4ceMzGEsj`

For a given version tag (e.g., `v0.1.0`), download **both** `benchmark_eval.jsonl.gz` and `SHA256SUMS.txt`.

### Verify download integrity
From the directory containing the downloaded files:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

### Load the benchmark
The artifact is JSONL (one JSON object per line):

```python
import gzip, json

path = "benchmark_eval.jsonl.gz"
with gzip.open(path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        # use ex...
        if i == 2:
            break
```

### Rebuild / update the benchmark
If you modify loaders or add sources, regenerate:

```bash
python3 curated_benchmark/scripts/normalize_datasets.py --out_dir curated_benchmark --include_hagrid
python3 curated_benchmark/scripts/package_release.py --version v0.1.0
```

### Notes on redistribution
This benchmark aggregates third-party datasets. Before sharing outside your group/org, check each source’s license/terms as documented in `curated_benchmark/dataset_card.md`.

