## Curated Benchmark: Evidence-Grounded QA Faithfulness & Robustness

### Task definition
This benchmark evaluates **faithfulness to provided evidence** for evidence-based / long-form question answering. Given a question and a set of evidence documents/passages, a model should produce an answer that is grounded in the evidence and avoids introducing unsupported claims.

Secondary focus: **robustness** under evidence-sensitive conditions (e.g., adversarial or perturbed contexts, distractor evidence, or instruction constraints when available).

### Included datasets (initial)
- **SynSciQA++** (`SynSciQA++.json`): inline evidence block in `instruction`; cited long-form answer in `response`.
- **CLAPNQ** (`curated_benchmark/raw/clapnq/annotated_data/...`): question + supporting passage(s) + long-form answer; includes selected supporting sentences (stored as `evidence_spans` text).
- **QuoteSum** (`curated_benchmark/raw/QuoteSum/v1/*.jsonl`): question + 1–8 Wikipedia source passages + multi-source summary answer.
- **HAGRID** (`curated_benchmark/raw/hagrid/hagrid-v1.0-en/*.jsonl`): question + quoted evidence passages (`quotes`) + one or more answers.
- **rag_uncertainty (claim-level)** (`curated_benchmark/raw/rag_uncertainty/claim_level/dataset/*.jsonl`): prompt + single source document + **model responses annotated with span-level faithfulness issues** (stored in `claims`).

### Excluded / not included (and why)
- **ExpertQA**: the downloaded repo files include questions, answers, and claim-level quality/attribution annotations, but (in the inspected JSONL files) do **not** include the underlying evidence passages as usable `documents`. Since this benchmark prioritizes evidence-grounded evaluation, we exclude it rather than fabricate evidence.
- **ELI5-WebGPT**: not included yet because the commonly referenced processed WebGPT/ELI5 evidence artifacts are not straightforwardly distributed as a ready-to-use evidence+answer dataset in the same way as the above sources. If you have a specific release file (or want to use a HuggingFace dataset snapshot), we can add a loader.

### Unified JSON schema
Each example is normalized to:

```json
{
  "id": "string",
  "source_dataset": "string",
  "question": "string",
  "documents": [
    {
      "doc_id": "string",
      "title": "string",
      "text": "string",
      "url": "string"
    }
  ],
  "gold_answer": "string",
  "evidence_spans": [],
  "claims": [],
  "labels": {
    "faithfulness": null,
    "factuality": null
  },
  "metadata": {
    "domain": "string",
    "answer_type": "long-form / short-form / claim-level",
    "num_documents": 0,
    "has_gold_evidence": false,
    "has_claim_labels": false,
    "has_perturbations": false,
    "source_instruction": "string"
  }
}
```

Notes:
- `documents` are **the provided evidence** (not retrieved at evaluation time).
- `evidence_spans` are included only when a dataset provides gold spans/citations in a machine-readable way. If not, we keep `documents` and set `has_gold_evidence=false`.
- `claims` is used for claim-level datasets or when an example includes explicit claim annotations.
- For `rag_uncertainty_claim_level`, `gold_answer` is a **model output to be evaluated**, not a human gold reference (this is intentional; the value is in the provided span-level issue annotations).

### Filtering rules
- Drop examples with missing/empty question.
- Drop examples with no usable evidence documents (empty document list after parsing/cleaning).
- Drop examples with empty/unusable gold answers.
- Do **not** drop examples missing evidence spans if evidence documents exist; instead mark `has_gold_evidence=false`.

### Limitations (expected)
- Some sources may not ship gold evidence spans or faithfulness labels.
- Mixing long-form QA and claim-level tasks requires careful analysis; incompatible datasets will be excluded and documented.
- Evidence spans may be **text-only** (e.g., CLAPNQ selected sentences) rather than character offsets into documents.

### Outputs produced by this build
- `curated_benchmark/benchmark_eval.json`: unified benchmark file.
- `curated_benchmark/stats.json`: basic dataset statistics.
- `curated_benchmark/scripts/inspect_datasets.py`: dataset sampler/field summarizer.
- `curated_benchmark/scripts/normalize_datasets.py`: normalizers/loaders and filtering rules.

