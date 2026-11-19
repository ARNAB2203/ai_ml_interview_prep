# Module 09 — Agent Frameworks: LangGraph, MCP, AutoGen, CrewAI, and ADK

**Level:** Advanced  
**Interview goal:** Select an orchestration style that makes state,
authorization, failure handling, and termination explicit.

## 1. Frameworks are orchestration tools, not security boundaries

An agent framework can simplify workflow composition, roles, state persistence,
or tool discovery. It cannot prove that a model-selected action is authorized.

```text
verified identity + tenant + quotas
              ↓
framework state / messages / tasks
              ↓
model-selected intent or handoff
              ↓
deterministic policy + typed tool gateway
              ↓
tool result + audit + evaluation trace
```

> **Warning:** Never grant a tool capability because a planner, role prompt, or
> “manager agent” says it is safe. Framework prose does not replace a policy
> decision based on verified identity and requested action.

## 2. Decision matrix

| Technology | Best fit | Fundamental concept to know | Production caveat |
| --- | --- | --- | --- |
| LangGraph | Durable, inspectable, conditional workflows | Typed state, nodes, edges, checkpoints, interrupts | State schema/version and idempotency still belong to your platform |
| MCP | Interoperable tool/resource/prompt servers | Tool discovery, JSON schema, transport/session, capability boundary | Authenticate/authorize at server; do not expose arbitrary shell/HTTP tools |
| AutoGen | Conversational multi-agent collaboration | Agent messages, handoff, termination, human input modes | Bound turns/tools; evaluate coordination failure and cost |
| CrewAI | Role/task/process-oriented teams | Crew, task, process, delegation, shared context | Role descriptions must not carry sensitive state or authorization |
| Google ADK | Agent/session/tool/evaluation workflow development | Agent, session, tool, runner, trace/evaluation | Treat model/provider/session data and tool auth as explicit boundaries |

**ADK note:** This module uses **Google Agent Development Kit** when it says
ADK. Confirm an interviewer’s meaning if the acronym is ambiguous.

## 3. LangGraph fundamentals

LangGraph is especially useful for agent systems that need visible transitions,
conditional routing, pause/resume, checkpointing, and replay. Prefer a graph to
an unbounded `while` loop when a task crosses approval, asynchronous, or
failure-recovery boundaries.

| State element | Example | Why it matters |
| --- | --- | --- |
| Immutable request metadata | tenant ID, correlation ID, agent version | Reproduction and safe propagation |
| Derived plan | selected retrieval/tool intent | Observable decision boundary |
| Approval state | trusted human/workflow decision | Prevents model text from impersonating approval |
| Retry/budget counters | attempts, tokens, deadline | Bounded execution and cost |
| Result/audit pointer | terminal outcome, trace ID | Supportability and evaluation |

### Concept → code: approval-gated LangGraph

```bash
python examples/framework_demos/langgraph_approval_graph.py
```

The graph routes a low-risk request directly to a bounded execution node. A
high-risk deployment request pauses unless trusted input sets
`approval_granted=True`. Read the trace: the action cannot execute simply
because the word “approved” appears in a model response.

For a real workflow, add a persistent checkpointer, a versioned state migration
plan, idempotency keys for action nodes, a time budget, and a resume mechanism
that re-authorizes the actor.

## 4. MCP fundamentals

Model Context Protocol provides a standardized way for a host to discover and
invoke tools, resources, and prompts exposed by a server. It does not turn an
untrusted tool into a safe one.

```text
MCP host / agent runtime
       │ discovery + typed invocation
       ▼
MCP server → authentication / tenant context → narrow business tool
                                               ↓
                                         downstream service
```

| MCP question | Senior answer |
| --- | --- |
| What should a tool expose? | A narrow typed business operation, not raw SQL, shell, filesystem, or unrestricted HTTP. |
| How is identity propagated? | Authenticated transport/session or delegated token; never a model-authored tenant field in production. |
| How do versions evolve? | Additive schemas, protocol/capability versioning, compatibility tests, deprecation windows. |
| How do you handle cancellation? | Deadline propagation, cancel-safe downstream calls, idempotent/compensating behavior, terminal audit event. |
| How do you isolate tenants? | Server-side authorization and data filtering; cache/telemetry/queue keys include tenant context. |

### Concept → code: real local FastMCP server

```bash
python examples/framework_demos/mcp_knowledge_server.py
# To host it over stdio for an MCP-capable client:
python examples/framework_demos/mcp_knowledge_server.py --stdio
```

The default command exercises actual FastMCP tool discovery and invocation
without a network connection. The server deliberately offers only
`lookup_runbook` and `calculate_capacity`; it rejects a mismatched subject and
tenant. In a real deployment, replace the visible demo fields with identity
derived from an authenticated session/transport.

## 5. AutoGen, CrewAI, and Google ADK: role/team patterns

Use a multi-agent framework when separate roles improve accuracy, isolation, or
review—not merely because a task can be split into prompts.

| Use case | Good bounded design | Bad design |
| --- | --- | --- |
| Incident triage | Analyzer retrieves redacted evidence; reviewer checks runbook citations; human approves remediation | Several agents all receive raw production credentials and chat indefinitely |
| Release review | Planner proposes a rollout; evaluator tests a fixed dataset; policy gate decides promotion | Manager agent trusts self-reported scores with no dataset/version provenance |
| Research synthesis | Parallel source summarizers return typed claims/citations; verifier rejects unsupported claims | Agents recursively delegate searches without budget or source controls |

The runnable [`multi_agent_patterns.py`](../examples/framework_demos/multi_agent_patterns.py)
does not pretend to be an SDK replacement. It teaches the portable contract:
role handoff, tenant propagation, maximum handoffs, explicit review, and trusted
approval. Those controls should remain in place whether the implementation uses
AutoGen, CrewAI, LangGraph, or Google ADK.

### Optional installation cells

```python
# Colab / VS Code — install only the framework being evaluated.
%pip install -q "autogen-agentchat>=0.7,<1" "autogen-ext[openai]>=0.7,<1"
# or
%pip install -q "crewai>=0.100,<2"
# or
%pip install -q "google-adk>=1,<2"
```

Provider-backed framework runs need a securely supplied provider credential.
Start with replay/fake clients and a fixed evaluation trace; do not make a
networked model call the first unit test.

## 6. Interview drill: multi-agent package deployment

**Prompt:** “Design a multi-agent workflow that validates and deploys a custom
agent package.”

Strong answer outline:

1. The upload/validator control plane creates an immutable artifact digest;
   no agent executes package code during validation.
2. A bounded analyzer can summarize manifest, scan, and evaluation results.
3. An evaluator checks policy and test thresholds against versioned evidence.
4. A LangGraph-style state machine pauses for approval if the target is
   production or the package has code/tool changes.
5. An MCP tool gateway exposes a narrow `request_deployment` operation that
   independently verifies tenant, artifact digest, scope, and approval.
6. Trace each handoff, impose turn/token/time budgets, and retain rollback.

## 7. Framework selection trade-offs

| Preference | Choose it when | Reconsider when |
| --- | --- | --- |
| Single typed service + explicit state machine | Workflow is modest and compliance/latency matter | You are rebuilding mature integration capabilities with no benefit |
| LangGraph | Long-lived/branching workflows need checkpoints and human interrupts | A simple one-step tool call becomes unnecessarily complex |
| AutoGen or CrewAI | Role decomposition is demonstrably useful and controlled | It becomes unbounded conversation with unclear ownership |
| ADK | Its ecosystem/session/evaluation model aligns with product/runtime choices | Portability or provider-neutral control-plane requirements dominate |
| MCP | A capability should be interoperable across hosts | The capability is too broad to safely publish as a generic tool |

Proceed to [Module 10: Model Engineering and Efficient Fine-tuning](10_model_engineering_and_efficient_finetuning.md).
