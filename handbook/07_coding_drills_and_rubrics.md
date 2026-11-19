# Module 07 — Coding Drills and Senior-Level Rubrics

**Level:** Beginner → Master  
**Interview goal:** Practice implementation under time constraints without
forgetting policy, tests, and operational behavior.

## 1. How to practice

For each drill, time-box the first implementation, then spend five minutes
describing what would change for production. Interviewers often value this
second part as much as a perfect algorithm.

| Duration | Focus |
| --- | --- |
| 20 minutes | Correct function, types, happy/edge cases, one or two tests |
| 45 minutes | Clear abstraction, validation, errors, test coverage |
| 90 minutes | API/service boundary, persistence/async design, security, observability |

## 2. Beginner drill — tokenizer with unknown handling

**Task:** Implement `encode(text)` and `decode(ids)` for a regex tokenizer.
Support punctuation and map unseen tokens to `<unk>`.

**Reference:** [`examples/tokenizer_bpe.py`](../examples/tokenizer_bpe.py)

| Rubric | Strong signal |
| --- | --- |
| Correctness | Round trips known text; does not crash on unseen words |
| Clarity | Separates tokenization from vocabulary lookup |
| Edge cases | Empty input, punctuation, case behavior, special tokens |
| Senior extension | Discusses tokenizer version/model compatibility and context budgets |

## 3. Intermediate drill — safe tool dispatcher

**Task:** Given a `ToolCall`, execute only registered tools authorized for the
tenant and return a typed error otherwise.

**Reference:** [`examples/tool_using_agent.py`](../examples/tool_using_agent.py)

Requirements:

1. Validate name and arguments before execution.
2. Check tenant/subject scope independently from the planner.
3. Retry only idempotent reads on transient errors.
4. Emit an audit record that does not contain secret values.

**Senior extension:** add deadlines, concurrency quotas, circuit breaking, and
human approval for non-idempotent effects.

## 4. Intermediate drill — grounded answer evaluator

**Task:** Given answer text, citations, and expected source IDs, compute a
retrieval/citation result and fail a quality gate if a cross-tenant source is
cited.

**Reference:** [`examples/rag_evaluation.py`](../examples/rag_evaluation.py)

Explain why citation presence is not enough: a citation can be irrelevant,
outdated, or inaccessible to the user. A production evaluator needs a curated
dataset and sampled human review.

## 5. Advanced drill — tenant-aware Agent API

**Task:** Implement create, list, get, update, and deploy endpoints for an
agent resource. The caller’s tenant comes from a verified boundary, never a
request body field.

**Reference:** [`examples/agent_platform_api.py`](../examples/agent_platform_api.py)

| Rubric | Strong signal |
| --- | --- |
| HTTP contract | Correct status codes, typed validation, idempotency/versioning discussion |
| Isolation | Cross-tenant reads return not found or forbidden consistently |
| State | Deployment allowed only from a validated state |
| Testability | In-process API tests plus policy tests |
| Production plan | DB-level isolation, transaction/outbox, auth integration, audit trail |

## 6. Advanced drill — validate a custom agent archive

**Task:** Validate a ZIP containing `agent.json` before any execution. Reject
path traversal, oversized archives, missing fields, unknown protocol versions,
and forbidden entrypoints.

**Reference:** [`examples/manifest_validator.py`](../examples/manifest_validator.py)

**Senior extension:** discuss signature verification, SBOM, malware scanning,
quarantine storage, asynchronous scanning, and a sandboxed runtime.

## 7. Master drill — design and defend a release gate

**Task:** Your team changes the agent prompt and a tool schema in the same
release. Define the CI/staging/canary gates and rollback conditions.

An excellent answer contains:

```text
versioned prompt + tool schema + code + evaluation set
      ↓
unit/schema/policy tests → offline quality & safety slices
      ↓
staging integration + load checks → canary by tenant/use case
      ↓
monitor task success, policy denials, tool errors, latency, cost
      ↓
automatic halt/rollback when explicit thresholds are breached
```

## 8. Self-review scorecard

After each answer, score 0–2.

| Criterion | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Problem framing | Jumps into code | Names one requirement | Clarifies outcome, constraints, and risk |
| Correctness | Happy path only | Some validation/tests | Explicit invariants and edge cases |
| Security | Not addressed | Mentions auth | Enforces identity, tenant scope, and least privilege |
| Operations | Not addressed | Mentions logs | Metrics, traces, rollout/rollback, ownership |
| Trade-offs | One asserted solution | Lists alternatives | Makes and defends a context-specific choice |

A senior target is 8/10 or better under the selected time limit.

For framework-specific drills, continue with [Module 08: Application Frameworks
and Provider APIs](08_application_frameworks_and_provider_apis.md).
