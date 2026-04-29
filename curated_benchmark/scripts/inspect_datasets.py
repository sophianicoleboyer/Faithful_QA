#!/usr/bin/env python3
"""
Inspect available datasets under curated_benchmark/raw and any explicitly listed files.

Outputs a human-readable summary:
- file paths
- detected format (json/jsonl/csv/tsv)
- top-level schema/fields
- 5–10 sampled examples with compact previews
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RAW_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "raw"


def _is_texty(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _compact_preview(x: Any, max_chars: int = 260) -> str:
    if x is None:
        return "null"
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, str):
        s = " ".join(x.split())
        if len(s) > max_chars:
            s = s[: max_chars - 3] + "..."
        return s
    if isinstance(x, list):
        return f"list(len={len(x)})"
    if isinstance(x, dict):
        keys = list(x.keys())
        head = ", ".join(map(str, keys[:12]))
        more = "" if len(keys) <= 12 else f", ...(+{len(keys)-12})"
        return f"dict(keys=[{head}{more}])"
    return f"{type(x).__name__}"


def _safe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_delimited(path: Path, delimiter: str) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield dict(row)


def _guess_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext == ".csv":
        return "csv"
    if ext == ".tsv":
        return "tsv"
    return "unknown"


def _sample_iter(it: Iterable[Any], k: int, seed: int) -> List[Any]:
    # Reservoir sampling for iterables with unknown length
    random.seed(seed)
    sample: List[Any] = []
    for i, item in enumerate(it):
        if i < k:
            sample.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def _summarize_dict_keys(samples: List[Any]) -> Tuple[List[str], Dict[str, int]]:
    key_counts: Dict[str, int] = {}
    for s in samples:
        if isinstance(s, dict):
            for k in s.keys():
                key_counts[k] = key_counts.get(k, 0) + 1
    keys_sorted = sorted(key_counts.keys(), key=lambda k: (-key_counts[k], k))
    return keys_sorted, key_counts


def inspect_file(path: Path, num_samples: int, seed: int) -> Dict[str, Any]:
    fmt = _guess_format(path)
    out: Dict[str, Any] = {"path": str(path), "format": fmt}

    try:
        if fmt == "json":
            obj = _safe_load_json(path)
            out["top_type"] = type(obj).__name__
            if isinstance(obj, list):
                out["num_records"] = len(obj)
                samples = obj[: min(num_samples, len(obj))]
            elif isinstance(obj, dict):
                # try to find a list-like field
                list_fields = [(k, v) for k, v in obj.items() if isinstance(v, list)]
                if list_fields:
                    k0, v0 = max(list_fields, key=lambda kv: len(kv[1]))
                    out["guessed_records_field"] = k0
                    out["num_records"] = len(v0)
                    samples = v0[: min(num_samples, len(v0))]
                else:
                    out["num_records"] = None
                    samples = [obj]
            else:
                out["num_records"] = None
                samples = [obj]

        elif fmt == "jsonl":
            it = _iter_jsonl(path)
            samples = _sample_iter(it, num_samples, seed)
            out["num_records"] = None
            out["top_type"] = "jsonl_stream"

        elif fmt in ("csv", "tsv"):
            delimiter = "," if fmt == "csv" else "\t"
            it = _iter_delimited(path, delimiter=delimiter)
            samples = _sample_iter(it, num_samples, seed)
            out["num_records"] = None
            out["top_type"] = f"{fmt}_stream"

        else:
            out["error"] = "unsupported_format"
            return out

        keys_sorted, key_counts = _summarize_dict_keys(samples)
        out["field_keys_ranked"] = keys_sorted[:60]
        out["field_presence_counts"] = {k: key_counts[k] for k in keys_sorted[:60]}

        out["samples"] = []
        for s in samples:
            if isinstance(s, dict):
                preview = {k: _compact_preview(s.get(k)) for k in list(s.keys())[:25]}
                out["samples"].append(preview)
            else:
                out["samples"].append(_compact_preview(s))

        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def discover_files(root: Path) -> List[Path]:
    exts = {".json", ".jsonl", ".csv", ".tsv"}
    paths: List[Path] = []
    if root.is_file() and root.suffix.lower() in exts:
        return [root]
    if not root.exists():
        return []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default=str(RAW_DIR_DEFAULT))
    ap.add_argument("--extra_files", type=str, nargs="*", default=[])
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    files = discover_files(raw_dir)
    for ef in args.extra_files:
        files.extend(discover_files(Path(ef).resolve()))
    files = sorted({p.resolve() for p in files})

    report = {
        "raw_dir": str(raw_dir),
        "num_files": len(files),
        "files": [],
    }

    for p in files:
        report["files"].append(inspect_file(p, num_samples=args.num_samples, seed=args.seed))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

