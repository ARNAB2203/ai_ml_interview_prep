# Module 14 — MLOps, Serving, Observability, and Responsible AI

**Level:** Advanced → Master
**Interview goal:** Operate an AI system as a versioned product with measurable
quality, safe deployment, owned failure modes, and a credible rollback plan.

## 1. From experiment to production artifact

```text
data snapshot + schema → feature/prompt/index code → experiment/evaluation
        ↓                         ↓                         ↓
immutable manifest ───────→ registry candidate ───────→ promotion gate
                                                              ↓
                                                 canary / monitor / rollback
```

An experiment is reproducible only when you can identify code, dependencies,
data/source snapshot, labels, feature or chunking logic, tokenizer/model/index,
configuration, seed, metrics, artifact digest, and the person/service that
promoted it.

| Artifact | Minimum provenance |
| --- | --- |
| Classical model | Training data/feature version, split strategy, hyperparameters, code, metrics/calibration |
| Fine-tuned LLM/adapter | Base revision, tokenizer, dataset, LoRA/quantization config, checkpoint, safety/task report |
| RAG index | Source/version, parser/chunker, embedding model, metadata schema, index build, deletion state |
| Prompt/agent | Prompt/tool/schema version, policy bundle, evaluation data, runtime model, trace/evidence |

## 2. ML pipelines and promotion gates

An idempotent pipeline makes safe restart possible:

```text
extract → validate → materialize features/index → train/build → evaluate
  → register immutable candidate → approve/promote → observe → rollback/retrain
```

| Pipeline property | How to implement it |
| --- | --- |
| Idempotency | Content-addressed inputs/outputs, operation IDs, safe upserts, no hidden mutable notebook state |
| Restartability | Persist stage output/metadata and resume only compatible stages |
| Promotion | Explicit quality, safety, latency, cost, and slice thresholds versus champion baseline |
| Rollback | Immutable version/alias switch; no manual reconstruction under incident pressure |
| Backfill/delete | Versioned replay, tenant filters, data-retention/deletion proof, downstream cache/index handling |

### Concept → code: local registry, promotion, drift, and privacy-safe traces

```bash
python examples/ml_ds_demos/mlops_lifecycle.py
```

The lab models an experiment registry/promotion decision, population stability
index (PSI) drift, segment behavior, and redacted trace attributes. It is a
small local counterpart to the contracts a managed registry or tracking system
such as MLflow, W&B, SageMaker, Vertex AI, or DVC should preserve.

> **Warning:** Drift is an investigation trigger, not an automatic retraining
> command. Verify source/data quality, segment impact, label delay, business
> changes, and actual task quality before promoting a replacement.

## 3. Model and LLM serving

| Serving mode | Favor when | Design concern |
| --- | --- | --- |
| Batch | Latency is not interactive; high throughput matters | Idempotency, partition/replay, late data, artifact version |
| Synchronous API | Short bounded predictions/answers | Deadline, concurrency, tail latency, back-pressure |
| Asynchronous job | Agent/retrieval/tool workflow can take time | Durable state, cancellation, result retention, status contract |
| Streaming | Partial response improves UX | Safe partial content, disconnect/cancel, token/back-pressure accounting |
| Embedded/edge | Data locality/latency/offline requirement | Artifact size, model updates, hardware/security constraints |

### Concept → code: FastAPI serving contract

```bash
python examples/ml_ds_demos/model_serving_api.py
```

The API creates a runtime in a lifespan hook and exposes `/healthz`, `/readyz`,
`/predict`, and Prometheus-style `/metrics`. It validates input, returns a
model version and request ID, and has in-process tests. Replace the deterministic
score with a verified model load only after defining artifact compatibility,
warm-up, memory/CPU/GPU limits, deadlines, and fallback behavior.

## 4. LLMOps performance and capacity

| Signal | Why it matters |
| --- | --- |
| Arrival rate and p95 duration | Little’s Law estimates in-flight work: `L = λ × W` |
| Queue depth / time | Back-pressure and user-visible saturation |
| Prefill/decode/token rate | Pinpoints model-runtime bottleneck |
| KV-cache memory | Limits concurrent long-context generation |
| Tool/retrieval latency | Agent p95 is often dominated by non-model work |
| Cost per successful task | Better than cost per request when quality varies |
| Cache hit rate | Can reduce repeated work but needs tenant/privacy-safe keys |

For GPU inference, select a serving platform such as vLLM, TGI, Triton,
KServe/BentoML, or a managed endpoint based on continuous batching, model
format, autoscaling, observability, tenancy, hardware, operational maturity,
and recovery needs—not brand preference.

## 5. Monitoring, observability, and incident response

```text
metrics: what changed?      traces: where/why did it change?
logs:   what evidence?      audit: who/what was authorized?
```

| Layer | Examples |
| --- | --- |
| Service RED | request rate, errors, duration; availability/error-budget impact |
| Resource USE | utilization, saturation, errors for CPU/GPU, queue, database, tool integration |
| ML | schema/data quality, prediction distribution, drift, delayed-label/slice quality, calibration |
| LLM/agent | model/prompt/tool/index version, token usage, retrieval/rerank, tool success, safety outcomes |
| Privacy | trace sampling, redaction, access control, retention/deletion, tenant-cost attribution |

OpenTelemetry-style spans should carry correlation ID, safe tenant pseudonym,
component/version, timing, error category, and cost/usage estimates. Never log
secrets or raw sensitive prompts/responses by default.

## 6. Responsible AI and model risk

| Control | Concrete practice |
| --- | --- |
| Intended use / limits | Model card with target users, non-goals, known failure modes, owner and review date |
| Fairness / slices | Define relevant segments, label quality, error costs, and acceptable gaps |
| Privacy / residency | Classify data, minimize payload, encrypt, retain/delete deliberately, control cross-region transfer |
| Safety / red team | Versioned adversarial cases: injection, jailbreak, PII, unsafe tool, harmful instruction, data exfiltration |
| Human oversight | Approval/escalation and appeal path proportional to decision impact |
| Governance | Risk register, change approval, evidence retention, incident communication, audit access |

Do not report “the model is unbiased” or “the agent is safe.” State the scope,
evaluation evidence, residual risk, monitoring, and fallback/human process.

## 7. Docker, CI/CD, cloud, and recovery checklist

| Layer | Required evidence |
| --- | --- |
| Container | Pinned base, minimal layers, non-root user, image scan/SBOM, health check, resource limits |
| CI | Unit/contract tests, dependency/supply-chain scan, schema/policy tests, offline model/agent evaluation |
| CD | Immutable artifact, IaC review, workload identity, secrets injection, canary, approval/rollback |
| Network | Private service paths, egress allowlist, tenant/database/vector-store access boundaries |
| Cloud | Service choice justified across ECS/EKS/Lambda/SageMaker/managed endpoint, queue, storage, DR |
| Resilience | RTO/RPO, backup/restore test, regional failure plan, runbook and ownership |

The repository includes a non-root, health-checked
[`model-serving.Dockerfile`](../deployments/model-serving.Dockerfile) and a
baseline GitHub Actions [quality gate](../.github/workflows/ai-quality.yml).
They are reference artifacts: adapt identity, base-image pinning, vulnerability
scan, secrets, IaC, environment promotion, and deployment approval to the
target organization rather than copying them blindly.

## 8. Portfolio standard and interview drill

For each portfolio project, include architecture, source/data/artifact lineage,
tests, evaluation, observability, security constraints, known limitations, and
an operational runbook. Three well-evidenced projects are stronger than many
notebooks with no failure model.

**Prompt:** “Offline evaluation is stable, but one customer segment degrades
after deployment. What do you do?”

1. Stabilize with canary halt/rollback if the segment breaches a defined risk.
2. Compare segment data schema/distribution, model/prompt/index/tool versions,
   latency and error traces, and delayed-label quality.
3. Reproduce with redacted traces and the correct tenant/slice; determine
   whether data, policy, retrieval, model, serving, or measurement changed.
4. Add the case to the regression set, repair the root cause, and promote only
   through the same evidence gate.

Proceed to [Module 15: Mathematical and Systems Interview Cheat Sheet](15_math_and_systems_interview_cheat_sheet.md).
