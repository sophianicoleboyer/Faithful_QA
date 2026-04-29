#!/usr/bin/env python3
"""
Create a shareable benchmark artifact for PolyBox (or similar):
- Convert benchmark_eval.json (list) -> benchmark_eval.jsonl
- Gzip to benchmark_eval.jsonl.gz
- Write SHA256SUMS.txt

This script does not modify the underlying dataset; it only packages it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "curated_benchmark"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_examples_from_json_list(path: Path) -> Iterable[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("benchmark_eval.json must be a JSON list")
    for ex in data:
        if isinstance(ex, dict):
            yield ex


def write_jsonl(examples: Iterable[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def gzip_file(in_path: Path, out_path: Path) -> None:
    with in_path.open("rb") as fin, gzip.open(out_path, "wb") as fout:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            fout.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=str, default="", help="Optional version tag (e.g., v0.1.0) for notes only.")
    ap.add_argument(
        "--in_json",
        type=str,
        default=str(BENCH_DIR / "benchmark_eval.json"),
        help="Input benchmark JSON (list)",
    )
    args = ap.parse_args()

    in_json = Path(args.in_json).resolve()
    if not in_json.exists():
        raise SystemExit(f"Missing input: {in_json}")

    out_jsonl = in_json.with_suffix(".jsonl")
    out_gz = out_jsonl.with_suffix(".jsonl.gz")
    sums_path = BENCH_DIR / "SHA256SUMS.txt"

    write_jsonl(iter_examples_from_json_list(in_json), out_jsonl)
    gzip_file(out_jsonl, out_gz)

    sha_gz = sha256_file(out_gz)
    sha_stats = sha256_file(BENCH_DIR / "stats.json") if (BENCH_DIR / "stats.json").exists() else ""

    lines: List[str] = []
    if args.version:
        lines.append(f"# version: {args.version}")
    lines.append(f"{sha_gz}  {out_gz.name}")
    if sha_stats:
        lines.append(f"{sha_stats}  stats.json")
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_gz}")
    print(f"Wrote {sums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

