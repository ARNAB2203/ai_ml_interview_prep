# Module 05 — Production Reliability, Security, and Deployment

**Level:** Advanced → Master  
**Interview goal:** Make an AI feature operable, governable, and recoverable in
production.

## 1. Reliability is designed at boundaries

Agent systems combine probabilistic generation with deterministic services.
Treat each external boundary as fallible: model provider, vector store, tool,
package registry, queue, deployment API, and identity provider.

| Failure | Control | Avoid |
| --- | --- | --- |
| Model timeout / 429 | Deadline, bounded retry with jitter, fallback or queue | Blind retries that multiply cost |
| Tool partial failure | Idempotency key, compensating action, durable state | Retrying a non-idempotent write |
| Bad model output | Schema validation, policy check, repair only within budget | Executing parsed text directly |
| Retrieval outage | Degrade to explicit “cannot ground answer” response | Inventing an answer without evidence |
| Queue redelivery | Idempotent consumer and deduplication | Assuming exactly-once delivery |
| Dependency vulnerability | SBOM, scan, patch policy, immutable rebuild | Modifying a running container |

## 2. Observability: traces answer “why?”

A request log is not sufficient for a multi-step run. Emit structured events
with a correlation ID and context that respects privacy policy.

| Signal | Example fields |
| --- | --- |
| Metrics | request count, p50/p95 latency, tool error rate, token/cost estimate, evaluation pass rate |
| Logs | event name, severity, correlation ID, tenant pseudonym, safe error code |
| Traces | agent run, model call, retrieval, tool calls, policy decision, deployment operation |
| Audit | actor, action, resource, authorization result, artifact/version, timestamp |

Never treat prompt or tool payload logging as harmless. Redact secrets and PII,
use access controls and retention periods, and sample deliberately.

## 3. Security and governance checklist

| Domain | Baseline control |
| --- | --- |
| Identity | OIDC/JWT verification, audience/issuer validation, short-lived credentials |
| Authorization | Tenant-scoped RBAC/ABAC and server-side enforcement for every tool |
| Secrets | Secret manager/workload identity; no credentials in prompts, images, or logs |
| Data | Classification, encryption, retention/deletion, tenant isolation, DLP where required |
| Prompt injection | Treat retrieved/user content as untrusted data; constrain tools and model instructions |
| Supply chain | Lockfiles, SBOM, image scan, signed artifacts, package policy |
| Runtime | Non-root container, read-only filesystem where possible, egress allowlist, CPU/memory limits |

### Prompt injection response pattern

1. Do not grant content the authority to alter system policy.
2. Keep sensitive instructions and secrets out of the model-accessible context
   where possible.
3. Validate every action using identity and deterministic policy—not a model’s
   assurance that it is safe.
4. Test adversarial documents and tool instructions in the evaluation suite.

## 4. Docker and AWS deployment pattern

```text
source + tests → immutable container image + SBOM → registry
       → policy/evaluation gate → canary deployment
       → health/readiness checks + telemetry → gradual rollout or rollback
```

| Stage | What to articulate |
| --- | --- |
| Build | Multi-stage image, pinned base, non-root user, unit/evaluation tests |
| Deploy | Immutable digest, environment-specific config, workload identity, secrets injection |
| Operate | Liveness versus readiness, autoscaling signal, quotas, dashboards, alerts |
| Recover | Versioned rollout, feature flag, index alias rollback, runbook, post-incident review |

On AWS, choose services based on workload rather than fashion: ECS/Fargate or
EKS for containerized long-running runtimes, Lambda for short bounded tasks,
SQS/EventBridge for asynchronous work, and managed databases with isolation
controls. State the reason, constraints, and rollback mechanism.

## 5. Evaluation becomes a release gate

Run deterministic API/unit tests and agent evaluation cases in CI. Separate
fast blocking checks from slower scheduled suites.

```text
Pull request: schema + unit + policy + small regression set
       ↓
Staging: integration + representative RAG/evaluation + load/safety slices
       ↓
Canary: production telemetry, error budget, human review where required
       ↓
Progress or automatic rollback
```

Record model, prompt, tool, index, dataset, and code versions together. Without
that provenance, a score regression cannot be explained or safely reversed.

## 6. Incident interview drill

**Prompt:** “Tool errors jumped after a model prompt update, but API latency is
normal. How do you respond?”

1. Stabilize: feature-flag or rollback the prompt/version if error budget is at
   risk; do not wait for perfect diagnosis.
2. Scope: compare tenant, tool, model, prompt, and request-type slices.
3. Reproduce: replay redacted traces against the prior and current version.
4. Diagnose: distinguish malformed tool selection, changed arguments, policy
   denial, provider behavior, or downstream tool failure.
5. Repair: add failing cases to the regression set, gate the change, canary
   again, and document the causal chain.

## 7. Master-level trade-offs to practice

| Tension | Thoughtful answer |
| --- | --- |
| Isolation vs cost | Start with logical isolation plus defense in depth; offer dedicated resources for regulated/high-risk tenants. |
| Latency vs quality | Budget each stage, retrieve/rerank conditionally, and expose an honest fallback rather than silent degradation. |
| Flexibility vs governance | Permit versioned configuration and custom packages behind validation, permissions, and approval workflows. |
| Release speed vs confidence | Use small deterministic gates, progressive delivery, and reliable rollback—not a single massive manual test. |
| Autonomy vs safety | Bound action scope and require approval proportional to impact. |

Proceed to [Module 06: System Design and Interview Drills](06_system_design_and_interview_drills.md).
