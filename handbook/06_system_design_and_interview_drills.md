# Module 06 — System Design and Interview Drills

**Level:** Master  
**Interview goal:** Lead a clear, constraint-driven architecture conversation
for the platform described in the JD.

## 1. Core system-design prompt

> Design a multi-tenant platform where customers create, test, deploy, and run
> AI agents that use approved knowledge bases and integration tools.

### A strong 45-minute answer structure

| Minute | Topic | What to cover |
| --- | --- | --- |
| 0–5 | Clarify | Tenant scale, data sensitivity, sync/async runs, tools, SLOs, cost, compliance |
| 5–12 | Control plane | Agent/version/deployment resources, API contracts, auth, state transitions |
| 12–20 | Data plane | Runtime, model gateway, tool gateway, RAG, streaming/cancellation |
| 20–28 | Isolation and security | Identity, tenant context, data/query/cache/telemetry boundaries |
| 28–35 | Reliability | Queues, idempotency, timeouts, rollout/rollback, failure paths |
| 35–42 | Quality | Evaluation datasets, gates, traces, online metrics, feedback loop |
| 42–45 | Trade-offs | MVP versus target state; risks and next validation |

### Reference architecture

```text
                         ┌──────────────── CONTROL PLANE ────────────────┐
Client ── OIDC/API GW ──► Agent API ─► config DB / artifact registry       │
             │              │                  │                           │
             │              └── evaluation / deployment workflow ─────────┤
             │                                                             │
             └──────── tenant + subject + scopes ─────────────────────────┘

                         ┌──────────────── DATA PLANE ───────────────────┐
Run request ────────────► Runtime / orchestrator ─► model gateway          │
                                  │          ├── retrieval service/index   │
                                  │          └── authorized tool gateway   │
                                  └── traces, audit, metrics                │
                         └────────────────────────────────────────────────┘
```

Say explicitly that the control plane manages desired state and authorization;
the data plane executes a versioned, approved artifact under resource and
network restrictions.

## 2. Capacity and cost discussion

Use equations to show structured thinking, then identify assumptions to measure.

```text
concurrent_runs ≈ requests_per_second × p95_run_seconds
monthly_model_cost ≈ runs × average_tokens_per_run × price_per_token
index_capacity ≈ chunks × vector_dimension × bytes_per_value × replication
```

Then explain admission control: per-tenant concurrency, token/tool budgets,
queues for non-interactive tasks, cache only when safe, and priority classes.

## 3. Deep-dive questions and answer signals

| Question | Strong answer signals |
| --- | --- |
| How would you make `POST /deployments` idempotent? | Client idempotency key, payload fingerprint, durable operation record, same terminal result on retry. |
| How do you isolate tenants in RAG? | Identity-derived tenant context, pre-filter/ACL at store, cache-key segregation, test for cross-tenant leaks. |
| How do you debug a bad agent answer? | Versioned trace from input policy through retrieved chunks, prompt/model/tool outputs, redacted replay, evaluation case. |
| How do you support custom code agents? | Quarantine, manifest/schema validation, scan, signed immutable artifact, sandbox, egress/resources, approval and revocation. |
| How do you release a prompt change? | Version it, offline regression/safety tests, canary by slice, monitor metrics, rollout/rollback. |
| Why not use one giant agent? | Separation of authority/data improves evaluation, least privilege, failure containment, and observability; coordination adds cost. |

## 4. Behavioral stories (STAR + engineering depth)

Prepare three stories. Keep the Situation/Task concise and spend time on
technical judgment, metrics, and what changed in the system.

| Story | Evidence to include |
| --- | --- |
| Quality regression | Baseline, failing slice, diagnosis, rollback, new regression gate |
| Cross-team API launch | Contract ownership, versioning, test environment, adoption and support plan |
| Reliability incident | Immediate mitigation, root cause, customer communication, permanent controls |

## 5. Whiteboard checklist

Before ending any system-design answer, verify that you stated:

- functional and non-functional requirements;
- API/resource contracts and async workflow behavior;
- tenant identity and authorization propagation;
- data lifecycle and sensitive-data boundaries;
- model/tool/RAG safety mechanisms;
- availability, latency, capacity, and cost assumptions;
- telemetry, evaluation, rollout, rollback, and ownership.

## 6. Ten rapid-fire interview prompts

1. Design a prompt/version registry with approval and rollback.
2. Add a new external tool to an existing agent platform safely.
3. Diagnose a retrieval regression after an embedding model migration.
4. Build a customer-visible API for asynchronous agent runs.
5. Migrate from shared to stronger tenant isolation for a regulated customer.
6. Explain how you would test a tool-calling agent without calling a live model.
7. Handle a tool that has eventual consistency and non-idempotent writes.
8. Build a package compatibility policy across runtime versions.
9. Define a practical evaluation program for weekly releases.
10. Reduce model cost 40% without harming a safety-critical workflow.

For each, start with the outcome and constraints, draw boundaries, then surface
the most meaningful trade-off.

Proceed to [Module 07: Coding Drills and Rubrics](07_coding_drills_and_rubrics.md).
