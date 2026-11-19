# Module 15 — Mathematical and Systems Interview Cheat Sheet

**Level:** Revision / Master
**Purpose:** Link equations to the engineering decision they support. Memorizing
symbols without stating the assumption or operational consequence is not a
senior-level answer.

## 1. ML and evaluation

| Concept | Formula / definition | Decision it informs |
| --- | --- | --- |
| Cosine similarity | `a·b / (||a|| ||b||)` | Normalize embeddings and avoid comparing vectors from incompatible models directly |
| MSE | `mean((y - ŷ)²)` | Penalize large regression errors; inspect outliers and scale |
| BCE | `-[y log p + (1-y) log(1-p)]` | Binary model objective and class weighting/calibration |
| Precision | `TP / (TP + FP)` | False-alert burden |
| Recall | `TP / (TP + FN)` | Missed-event burden |
| F1 | `2PR / (P + R)` | A balance only when false positives/negatives are similarly important |
| ROC-AUC | Ranking across all thresholds | General separability; can mislead on rare positive class |
| PR-AUC | Precision-recall ranking | Imbalanced-event ranking quality |
| Brier score | `mean((p - y)²)` | Probability calibration |
| Regularization | L1 / L2 penalty | Sparsity versus smooth coefficient shrinkage |

## 2. Retrieval and RAG

| Concept | Formula / definition | Decision it informs |
| --- | --- | --- |
| Recall@k | Expected source appears in top-k | Candidate retrieval coverage |
| MRR | Mean `1 / rank(first relevant)` | How soon evidence appears |
| NDCG | Discounted relevance divided by ideal ranking | Graded top-rank relevance |
| Reciprocal rank fusion | `Σ 1 / (k + rank_i)` | Combine sparse and dense rankings robustly |
| Context precision | Supporting chunks / retrieved chunks | Prompt/context waste and citation quality |
| Freshness | source/index version age | Re-index/alias/rollback decision |

Remember: a strong final answer does not prove the retriever was correct. Keep
retrieval metrics, citation validation, groundedness, and task success distinct.

## 3. Transformers and parameter-efficient tuning

| Concept | Formula / definition | Decision it informs |
| --- | --- | --- |
| Attention | `softmax((QKᵀ / √d_k) + mask)V` | Shape/mask correctness and context cost |
| Causal mask | Future-token score = `-∞` before softmax | Prevent generation-time information leakage |
| LoRA | `W' = W + scale × BA` | Adapter capacity/rank/target-module choice |
| QLoRA | Quantized frozen base + trainable LoRA adapters | VRAM reduction versus GPU/library/quality constraints |
| KV cache | approximately K+V × layers × hidden width × tokens × batch | Concurrent long-context serving capacity |

## 4. Systems and capacity

| Concept | Formula / definition | Decision it informs |
| --- | --- | --- |
| Little’s Law | `L = λW` | In-flight concurrency from request arrival rate and p95 duration |
| Global batch | micro-batch × accumulation × data-parallel world size | Training stability/capacity plan |
| Throughput | completed work / time | Separate from individual request latency |
| p95/p99 | Tail of latency distribution | SLO, queue/back-pressure, user experience |
| Error budget | allowed unreliable time/requests in an SLO period | Release throttle/rollback decision |
| RTO/RPO | recovery time / acceptable data loss | Disaster recovery architecture |

## 5. A four-level answer template

| Level | What to include |
| --- | --- |
| Basic | Definition plus a concrete example |
| Intermediate | Trade-off and selection criterion |
| Advanced | Failure mode, measurement, and mitigation |
| Scenario | Clarifying requirements, baseline, metrics, rollout, rollback, and ownership |

### Example: “How do you reduce LLM cost?”

1. Define successful task and quality/safety floor before optimizing spend.
2. Break cost down by model, input/output tokens, retries, retrieval, tool calls,
   and tenant/use-case slice.
3. Try one measured change at a time: cache safe results, reduce context,
   improve retrieval/rerank, route simple work to a smaller model, batch where
   latency permits, or tune output budget.
4. Run offline regression/safety checks, canary, monitor cost per successful
   task and p95, then rollback if critical quality degrades.

## 6. Final revision checklist

- Can I explain the data/label/source split before naming a model?
- Can I draw the tenant/identity/tool/retrieval boundaries before naming a
  framework?
- Can I state exactly what metric would block a release?
- Can I distinguish model, data, serving, policy, and measurement failures?
- Can I name the immutable version/artifact and rollback action?
- Can I say what evidence I would collect before changing production?

Return to [the overview and study plan](00_overview_and_study_plan.md).
