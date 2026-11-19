# Module 03 — RAG, Grounding, and Evaluation

**Level:** Intermediate → Advanced  
**Interview goal:** Show that retrieval quality is measurable, tenant-aware, and
not confused with fluent generation.

## 1. RAG pipeline

Retrieval-augmented generation (RAG) augments a response with selected source
content. It does not make a model intrinsically truthful.

```text
authorized query
      ↓
query normalization → tenant/ACL filter → candidate retrieval → rerank
                                                           ↓
response synthesis ← cited evidence ← context budget / prompt construction
      ↓
answer, source IDs, retrieval metrics, trace
```

Filtering by tenant and document ACL must occur before—or be enforced by—the
retrieval store. Filtering a global top-k result afterward can leak the fact
that sensitive data exists and can degrade results unpredictably.

## 2. Chunking and indexing choices

| Choice | Benefit | Trade-off |
| --- | --- | --- |
| Fixed-size chunks | Simple, predictable cost | Splits tables or semantic units |
| Structure-aware chunks | Better citation and context | Requires parsers and metadata |
| Dense retrieval | Handles lexical variation | Can miss exact IDs and negation |
| Keyword retrieval | Strong for exact terms and filters | Weak semantic paraphrase matching |
| Hybrid retrieval + rerank | Often best quality | More latency and operational complexity |

Attach `tenant_id`, source/version, ACL, timestamp, document type, and chunk
position as metadata. Those fields support authorization, freshness rules,
citations, deletion, and debugging.

## 3. Concept → code: local retrieval and an evaluation gate

```bash
python examples/rag_evaluation.py
```

The example deliberately substitutes a deterministic hashing embedder for a
cloud embedding model. It demonstrates the production seam without requiring a
network or secret:

1. Index small tenant-tagged documents.
2. Retrieve only documents belonging to the request tenant.
3. Return answer text with source IDs.
4. Run a versioned evaluation set and fail if grounding, retrieval, or latency
   thresholds are violated.

> **Expected observation:** A cross-tenant document with similar wording is
> never returned. If you remove the tenant filter in a local experiment, the
> test should fail—this is a security regression, not merely a quality change.

## 4. What to evaluate

| Metric | Question it answers | Typical method |
| --- | --- | --- |
| Retrieval recall@k | Did the evidence appear in candidates? | Gold source IDs in an offline set |
| Citation precision | Do cited sources support the answer? | Human labels, rules, or judge model with calibration |
| Groundedness | Is the answer entailed by retrieved context? | Claim-to-evidence review |
| Task success | Did the user’s intended job finish correctly? | Scenario-based checks and product telemetry |
| Safety | Did the agent refuse / avoid forbidden actions? | Adversarial cases and policy tests |
| p95 latency / cost | Is quality sustainable in production? | Traces partitioned by tenant and route |

Do not use an LLM judge as the only truth source. Calibrate it against human
labels, keep prompt/model versions, and inspect disagreements by slice.

## 5. Evaluation dataset discipline

A reusable evaluation set is a small product:

- version it with source snapshot IDs and expected behavior;
- include normal, ambiguous, adversarial, stale, and no-answer cases;
- avoid leaking it into model fine-tuning or prompt optimization loops;
- record per-case failure reasons, not just one aggregate score; and
- make thresholds explicit and compare to a baseline, not wishful targets.

### Example quality gate

```text
release allowed when:
  grounded_answer_rate >= 0.95
  retrieval_recall_at_2 >= 0.95
  cross_tenant_leaks == 0
  p95_retrieval_latency_ms <= 250
  no critical policy test regresses
```

Thresholds are product decisions. A medical or financial workflow may require
human review and much stricter gates; an internal discovery tool may prioritize
recall and label uncertainty.

## 6. Interview drill: answer-with-citations service

**Prompt:** “Customers report answers cite outdated policy documents. What do
you do?”

Strong response:

1. Reproduce with the trace ID, tenant, query, index version, prompt version,
   retrieved chunk IDs, and model version.
2. Determine whether the fault is ingestion freshness, metadata filtering,
   retrieval/reranking, prompt construction, or citation rendering.
3. Add a temporal validity filter and reindex only the affected source lineage.
4. Add the reported cases to the offline evaluation set.
5. Canary the fix, inspect slice metrics, and retain a rollbackable index alias.

Proceed to [Module 04: Platform API, Multi-tenancy, and Packaging](04_platform_api_multitenancy_and_packaging.md).
