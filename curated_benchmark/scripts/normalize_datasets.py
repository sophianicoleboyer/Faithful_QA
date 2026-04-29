#!/usr/bin/env python3
"""
Normalize supported datasets into a unified evidence-grounded QA schema.

Currently supported:
- SynSciQA++ (local file: SynSciQA++.json)
- rag_uncertainty claim-level (curated_benchmark/raw/rag_uncertainty/claim_level/dataset)
- CLAPNQ (curated_benchmark/raw/clapnq/annotated_data)
- QuoteSum (curated_benchmark/raw/QuoteSum/v1)

Notes:
- ExpertQA is downloaded for inspection, but appears to ship attribution/claim labels without
  distributing full evidence passages in the repo files we inspected; we therefore do not
  normalize it as evidence-grounded QA (no usable `documents`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = DEFAULT_OUT_DIR / "raw"


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def normalize_whitespace(s: str) -> str:
    return " ".join(s.split()).strip()


def parse_synsciqa_instruction(instruction: str) -> Tuple[str, List[Dict[str, str]], str]:
    """
    Returns: (question, documents, remainder_after_sources)

    The instruction contains:
      [BEGIN OF SOURCES] ... [END OF SOURCES]
    and includes question text as:
      question "...."
    """
    m = re.search(r"\[BEGIN OF SOURCES\](.*)\[END OF SOURCES\](.*)$", instruction, flags=re.S)
    if not m:
        raise ValueError("Missing [BEGIN/END OF SOURCES] markers")
    sources_blob = m.group(1).strip()
    rest = m.group(2).strip()

    qm = re.search(r'question\s+"([^"]+)"', rest, flags=re.I)
    if not qm:
        raise ValueError('Missing question "..." pattern')
    question = normalize_whitespace(qm.group(1))

    # Split sources into documents. Typical line:
    #   Cooley & Mitchell, 2015, p. 123: Proteomics, ...
    # There may be multi-line passages; the blob uses newline separators.
    # We split on occurrences of a "citation header:" at line starts.
    pattern = re.compile(r"(?m)^(?P<header>[^:\n]{3,200}?:)\s*")
    starts = [(m.start(), m.end(), m.group("header")[:-1].strip()) for m in pattern.finditer(sources_blob)]
    documents: List[Dict[str, str]] = []
    if not starts:
        # fallback: treat entire blob as single document
        documents.append(
            {
                "doc_id": f"synsciqa_doc_{stable_hash(sources_blob)}",
                "title": "",
                "text": normalize_whitespace(sources_blob),
                "url": "",
            }
        )
    else:
        for idx, (s0, e0, header) in enumerate(starts):
            s_text = e0
            e_text = starts[idx + 1][0] if idx + 1 < len(starts) else len(sources_blob)
            passage = sources_blob[s_text:e_text].strip()
            doc_id = f"synsciqa_doc_{stable_hash(header + '|' + passage)}"
            documents.append(
                {
                    "doc_id": doc_id,
                    "title": normalize_whitespace(header),
                    "text": normalize_whitespace(passage),
                    "url": "",
                }
            )

    return question, documents, rest


def is_usable_text(s: Optional[str]) -> bool:
    return isinstance(s, str) and normalize_whitespace(s) != ""


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_synsciqa_plus_plus(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("SynSciQA++.json must be a JSON list")

    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            continue
        instruction = ex.get("instruction", "")
        response = ex.get("response", "")

        # filtering rules
        if not is_usable_text(instruction):
            continue
        if not is_usable_text(response):
            continue

        try:
            question, documents, _rest = parse_synsciqa_instruction(instruction)
        except Exception:
            continue

        documents = [d for d in documents if is_usable_text(d.get("text"))]
        if not question:
            continue
        if not documents:
            continue

        example_id = f"synsciqa++_{i:05d}"
        out.append(
            {
                "id": example_id,
                "source_dataset": "SynSciQA++",
                "question": question,
                "documents": documents,
                "gold_answer": normalize_whitespace(response),
                "evidence_spans": [],
                "claims": [],
                "labels": {"faithfulness": None, "factuality": None},
                "metadata": {
                    "domain": "science",
                    "answer_type": "long-form",
                    "num_documents": len(documents),
                    "has_gold_evidence": False,
                    "has_claim_labels": False,
                    "has_perturbations": False,
                    "source_instruction": instruction,
                },
            }
        )

    return out


def normalize_rag_uncertainty_claim_level(dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Paper 8 (rag_uncertainty) claim-level dataset.

    - `source_info.jsonl`: prompt + source document text
    - `response.jsonl`: model response + span-level labels
    """
    src_path = dataset_dir / "source_info.jsonl"
    resp_path = dataset_dir / "response.jsonl"
    if not src_path.exists() or not resp_path.exists():
        return []

    sources: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(src_path):
        sid = str(row.get("source_id", "")).strip()
        if not sid:
            continue
        sources[sid] = row

    out: List[Dict[str, Any]] = []
    for row in iter_jsonl(resp_path):
        sid = str(row.get("source_id", "")).strip()
        src = sources.get(sid)
        if not src:
            continue

        question = src.get("prompt", "")
        evidence_text = src.get("source_info", "")
        response = row.get("response", "")
        if not is_usable_text(question) or not is_usable_text(evidence_text) or not is_usable_text(response):
            continue

        doc_id = f"rag_uncertainty_source_{stable_hash(sid)}"
        documents = [
            {
                "doc_id": doc_id,
                "title": str(src.get("source", "")),
                "text": str(evidence_text),
                "url": "",
            }
        ]

        claims: List[Dict[str, Any]] = []
        labs = row.get("labels") or []
        if isinstance(labs, list):
            for j, lab in enumerate(labs):
                if not isinstance(lab, dict):
                    continue
                claims.append(
                    {
                        "claim_id": f"rag_uncertainty_claim_{stable_hash(sid + '|' + str(row.get('id','')) + '|' + str(j))}",
                        "text": str(lab.get("text", "")),
                        "offsets": {"start": lab.get("start"), "end": lab.get("end")},
                        "label": str(lab.get("label_type", "")),
                        "meta": str(lab.get("meta", "")),
                        "doc_id": doc_id,
                    }
                )

        example_id = f"rag_uncertainty_{sid}_{row.get('id')}_{stable_hash(str(row.get('model','')))}"
        out.append(
            {
                "id": example_id,
                "source_dataset": "rag_uncertainty_claim_level",
                "question": normalize_whitespace(str(question)),
                "documents": documents,
                # This is a model output annotated for faithfulness issues.
                "gold_answer": str(response),
                "evidence_spans": [],
                "claims": claims,
                "labels": {"faithfulness": None, "factuality": None},
                "metadata": {
                    "domain": "news/summarization",
                    "answer_type": "claim-level",
                    "num_documents": 1,
                    "has_gold_evidence": False,
                    "has_claim_labels": bool(claims),
                    "has_perturbations": False,
                    "source_instruction": "",
                    "source_id": sid,
                    "split": str(row.get("split", "")),
                    "model": str(row.get("model", "")),
                    "temperature": row.get("temperature", None),
                    "quality": str(row.get("quality", "")),
                },
            }
        )

    return out


def normalize_clapnq(annotated_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not annotated_dir.exists():
        return out

    for split in ("train", "dev"):
        split_dir = annotated_dir / split
        if not split_dir.exists():
            continue
        for fname in sorted(split_dir.glob("*.jsonl")):
            for row in iter_jsonl(fname):
                q = row.get("input", "")
                if not is_usable_text(q):
                    continue
                passages = row.get("passages") or []
                if not isinstance(passages, list) or not passages:
                    continue

                documents: List[Dict[str, str]] = []
                for di, p in enumerate(passages):
                    if not isinstance(p, dict):
                        continue
                    text = p.get("text", "")
                    if not is_usable_text(text):
                        continue
                    title = str(p.get("title", ""))
                    documents.append(
                        {
                            "doc_id": f"clapnq_{stable_hash(str(row.get('id','')))}_{di}",
                            "title": normalize_whitespace(title),
                            "text": str(text),
                            "url": "",
                        }
                    )
                if not documents:
                    continue

                output = row.get("output") or []
                gold_answer = ""
                evidence_spans: List[Dict[str, Any]] = []
                if isinstance(output, list) and output and isinstance(output[0], dict):
                    gold_answer = output[0].get("answer", "") or ""
                    selected = output[0].get("selected_sentences") or []
                    if isinstance(selected, list):
                        for s in selected:
                            if is_usable_text(s):
                                evidence_spans.append({"doc_id": documents[0]["doc_id"], "text": str(s)})
                if not is_usable_text(gold_answer):
                    continue

                ex_id = f"clapnq_{split}_{row.get('id', stable_hash(q))}"
                out.append(
                    {
                        "id": ex_id,
                        "source_dataset": "CLAPNQ",
                        "question": str(q),
                        "documents": documents,
                        "gold_answer": normalize_whitespace(str(gold_answer)),
                        "evidence_spans": evidence_spans,
                        "claims": [],
                        "labels": {"faithfulness": None, "factuality": None},
                        "metadata": {
                            "domain": "open-domain",
                            "answer_type": "long-form",
                            "num_documents": len(documents),
                            "has_gold_evidence": bool(evidence_spans),
                            "has_claim_labels": False,
                            "has_perturbations": False,
                            "source_instruction": "",
                            "split": split,
                            "file": fname.name,
                        },
                    }
                )

    return out


def normalize_quotesum(v1_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not v1_dir.exists():
        return out

    for split in ("train", "dev", "test"):
        path = v1_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            q = row.get("question", "")
            a = row.get("summary", "")
            if not is_usable_text(q) or not is_usable_text(a):
                continue

            documents: List[Dict[str, str]] = []
            uid = str(row.get("unique_id", "")) or stable_hash(str(q))
            for i in range(1, 9):
                text = row.get(f"source{i}", "")
                title = row.get(f"title{i}", "")
                if not is_usable_text(text):
                    continue
                documents.append(
                    {
                        "doc_id": f"quotesum_{stable_hash(uid)}_{i}",
                        "title": normalize_whitespace(str(title)),
                        "text": str(text),
                        "url": "",
                    }
                )
            if not documents:
                continue

            ex_id = f"quotesum_{split}_{uid}"
            out.append(
                {
                    "id": ex_id,
                    "source_dataset": "QuoteSum",
                    "question": str(q),
                    "documents": documents,
                    "gold_answer": normalize_whitespace(str(a)),
                    "evidence_spans": [],
                    "claims": [],
                    "labels": {"faithfulness": None, "factuality": None},
                    "metadata": {
                        "domain": "wikipedia",
                        "answer_type": "long-form",
                        "num_documents": len(documents),
                        "has_gold_evidence": False,
                        "has_claim_labels": False,
                        "has_perturbations": False,
                        "source_instruction": "",
                        "qid": str(row.get("qid", "")),
                        "split": split,
                        "short_answers": [
                            row.get(f"short_ans_{i}")
                            for i in range(1, 9)
                            if is_usable_text(row.get(f"short_ans_{i}"))
                        ],
                        "covered_short_answers": row.get("covered_short_answers", None),
                    },
                }
            )

    return out


def normalize_hagrid(hagrid_dir: Path) -> List[Dict[str, Any]]:
    """
    HAGRID JSONL (downloaded from HuggingFace as raw files):
      - query: question
      - quotes: [{docid, idx, text}, ...]  (evidence passages)
      - answers: [{answer, answer_type, informative, sentences}, ...]

    We convert each (query, answer) pair into a separate example, reusing the same evidence docs.
    """
    out: List[Dict[str, Any]] = []
    if not hagrid_dir.exists():
        return out

    for split in ("train", "dev"):
        path = hagrid_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            q = row.get("query", "")
            if not is_usable_text(q):
                continue

            quotes = row.get("quotes") or []
            documents: List[Dict[str, str]] = []
            if isinstance(quotes, list):
                for qi, qt in enumerate(quotes):
                    if not isinstance(qt, dict):
                        continue
                    txt = qt.get("text", "")
                    if not is_usable_text(txt):
                        continue
                    documents.append(
                        {
                            "doc_id": f"hagrid_{stable_hash(str(row.get('query_id','')))}_{qi}",
                            "title": normalize_whitespace(str(qt.get("docid", ""))),
                            "text": str(txt),
                            "url": "",
                        }
                    )
            if not documents:
                continue

            answers = row.get("answers") or []
            if not isinstance(answers, list) or not answers:
                continue
            for ai, ans in enumerate(answers):
                if not isinstance(ans, dict):
                    continue
                gold = ans.get("answer", "")
                if not is_usable_text(gold):
                    continue
                ex_id = f"hagrid_{split}_{row.get('query_id','')}_{ai}"
                out.append(
                    {
                        "id": ex_id,
                        "source_dataset": "HAGRID",
                        "question": str(q),
                        "documents": documents,
                        "gold_answer": normalize_whitespace(str(gold)),
                        "evidence_spans": [],
                        "claims": [],
                        "labels": {"faithfulness": None, "factuality": None},
                        "metadata": {
                            "domain": "open-domain",
                            "answer_type": "long-form",
                            "num_documents": len(documents),
                            "has_gold_evidence": False,
                            "has_claim_labels": False,
                            "has_perturbations": False,
                            "source_instruction": "",
                            "split": split,
                            "query_id": row.get("query_id", None),
                            "answer_type_detail": ans.get("answer_type", None),
                            "informative": ans.get("informative", None),
                        },
                    }
                )

    return out


def compute_stats(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        by_source.setdefault(ex.get("source_dataset", "unknown"), []).append(ex)

    def avg(nums: List[int]) -> float:
        return float(sum(nums)) / float(len(nums)) if nums else 0.0

    stats: Dict[str, Any] = {"total_examples": len(examples), "by_source": {}}
    for src, exs in sorted(by_source.items(), key=lambda kv: kv[0]):
        num_docs = [len(e.get("documents") or []) for e in exs]
        ans_lens = [len((e.get("gold_answer") or "").split()) for e in exs]
        with_spans = sum(1 for e in exs if (e.get("evidence_spans") or []))
        with_claims = sum(1 for e in exs if (e.get("claims") or []))
        with_labels = sum(
            1
            for e in exs
            if (e.get("labels") or {}).get("faithfulness") is not None
            or (e.get("labels") or {}).get("factuality") is not None
        )
        stats["by_source"][src] = {
            "num_examples": len(exs),
            "avg_num_documents": avg(num_docs),
            "avg_answer_length_words": avg(ans_lens),
            "num_with_evidence_spans": with_spans,
            "num_with_claim_labels": with_claims,
            "num_with_faithfulness_or_factuality_labels": with_labels,
        }

    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--synsciqa_path",
        type=str,
        default=str(ROOT / "SynSciQA++.json"),
        help="Path to SynSciQA++.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory (default: curated_benchmark/)",
    )
    ap.add_argument(
        "--include_hagrid",
        action="store_true",
        help="Include locally downloaded HAGRID JSONL from curated_benchmark/raw/hagrid/hagrid-v1.0-en/.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    examples: List[Dict[str, Any]] = []
    syn_path = Path(args.synsciqa_path).resolve()
    if syn_path.exists():
        examples.extend(normalize_synsciqa_plus_plus(syn_path))

    ru_dir = RAW_DIR / "rag_uncertainty" / "claim_level" / "dataset"
    if ru_dir.exists():
        examples.extend(normalize_rag_uncertainty_claim_level(ru_dir))

    clap_dir = RAW_DIR / "clapnq" / "annotated_data"
    if clap_dir.exists():
        examples.extend(normalize_clapnq(clap_dir))

    qs_dir = RAW_DIR / "QuoteSum" / "v1"
    if qs_dir.exists():
        examples.extend(normalize_quotesum(qs_dir))

    if args.include_hagrid:
        hg_dir = RAW_DIR / "hagrid" / "hagrid-v1.0-en"
        if hg_dir.exists():
            examples.extend(normalize_hagrid(hg_dir))
        else:
            print(
                f"[warn] HAGRID dir missing: {hg_dir}. Download train/dev JSONL into this folder first.",
                file=sys.stderr,
            )

    # save outputs
    (out_dir / "benchmark_eval.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stats = compute_stats(examples)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

