# Module 02 — Agent Design, Tool Use, and Orchestration

**Level:** Intermediate  
**Interview goal:** Design an agent that is useful without granting it
unbounded authority.

## 1. An agent is a controlled workflow, not a magical loop

For platform work, define an agent as a policy-controlled workflow that uses a
model to select or synthesize actions within explicit limits.

```text
request + identity + tenant context
             ↓
input validation → policy / budget check → planner
                                      ↓
                             structured tool call
                                      ↓
                         authorization + schema validation
                                      ↓
                          tool execution / result normalization
                                      ↓
                      response + audit event + evaluation trace
```

The model is one component. Identity, authorization, input validation, retry
rules, and auditability must remain deterministic platform controls.

## 2. Prompting and structured outputs

Prompts should be versioned configuration, not anonymous strings scattered
through code. A robust system instruction specifies role, allowed tools,
response schema, refusal behavior, data boundaries, and a maximum-step budget.

| Pattern | Useful when | Failure mode to discuss |
| --- | --- | --- |
| Free-form text response | Conversational explanation | Hard to parse; can hide unsupported claims |
| JSON-schema response | API workflows or tool selection | Schema-valid content can still be unsafe or wrong |
| Function/tool call | Action requires deterministic parameters | Model can choose an unauthorized or costly action |
| Planner–executor | Multi-step task needs transparent state | More latency, more traces, and failure propagation |

> **Warning:** Validate a model-produced JSON object twice: first against its
> schema, then against policy. A valid `{"amount": 1000000}` is not
> automatically an authorized transfer.

## 3. Tool design checklist

Every tool contract should state:

- its input/output schema and version;
- required tenant, subject, and scopes;
- idempotency behavior and side-effect classification;
- timeout, retry, and circuit-breaker policy;
- redaction and audit requirements; and
- cost/rate limits and a user-confirmation boundary for consequential actions.

Avoid exposing an unrestricted shell, database connection, or HTTP client as a
tool. Offer narrow business operations instead, such as `get_invoice_status`
or `search_authorized_knowledge`, and enforce their policy server-side.

## 4. Concept → code: deterministic tool-using agent

```bash
python examples/tool_using_agent.py
```

The script intentionally uses a rule-based planner rather than a hosted LLM.
That makes the important components repeatable:

| Component | What to inspect in code |
| --- | --- |
| `TenantContext` | Tenant and subject follow every execution path. |
| `ToolSpec` / `ToolCall` | The tool name and arguments are structured. |
| `PolicyEngine` | Authorization and input constraints are separate from planning. |
| `ToolRunner` | Transient failures retry only for an idempotent read. |
| `AuditEvent` | Decision, outcome, and timing are recorded without secret data. |

Expected output includes a successful calculation, an authorized knowledge
lookup, a denied cross-tenant request, and audit records. Replace the planner
with an LLM adapter only after these boundaries exist.

## 5. State, memory, and multi-step workflows

Use a state machine for workflows with material transitions. Persist only the
minimum state required for recovery and protect it with tenant-aware access.

| State | Example transition | Invariant |
| --- | --- | --- |
| `draft` | User saves agent configuration | Configuration validates against its schema |
| `testing` | Evaluation run begins | Runtime version and test set are recorded |
| `approved` | Gate passes / human approves | Policy and deployment target are authorized |
| `deployed` | Artifact activates | Immutable artifact digest is known |
| `suspended` | Risk or budget breach | New runs are blocked deterministically |

For long-running workflows, favor durable jobs and an event/outbox pattern over
holding an HTTP request open. Propagate a correlation ID across planner, tools,
retrieval, and deployment so support can reconstruct a run.

## 6. Interview drill: autonomous remediation agent

**Prompt:** “Design an agent that reads failed deployment logs and proposes a
remediation.”

Answer in this order:

1. Clarify whether it can only recommend or can make changes.
2. Ingest tenant-scoped, redacted logs and preserve retention limits.
3. Retrieve approved runbooks; cite them in a structured recommendation.
4. Allow only read-only diagnostics by default.
5. Require an approval token and an idempotent, narrow tool for changes.
6. Evaluate false-positive recommendations, unsafe tool attempts, time to
   diagnose, and per-tenant cost.

## 7. Production trade-offs

| Decision | Favor it when | Cost |
| --- | --- | --- |
| Single agent loop | Task is simple and actions are few | Limited specialization |
| Planner–executor | Steps need inspection and retries | Extra latency and more state |
| Multi-agent delegation | Roles/data boundaries are genuinely distinct | Coordination failures and harder evaluation |
| Human approval | Action is irreversible or high impact | Slower completion |
| Fully automatic tool call | Action is low risk, idempotent, and tightly bounded | Must still monitor drift and abuse |

Proceed to [Module 03: RAG and Evaluation](03_rag_and_evaluation.md).
