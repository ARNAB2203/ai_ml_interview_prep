# Module 08 — Application Frameworks, Provider APIs, and FastAPI

**Level:** Intermediate → Advanced  
**Interview goal:** Choose and integrate a framework without surrendering
security, tenancy, evaluation, or portability to that framework.

## 1. The framework stack in context

The names in this module solve different layers of an AI application. A strong
candidate starts with the boundary and requirement, then selects a framework.

```text
FastAPI / API gateway                 → HTTP contract, identity, operations
Provider SDK (OpenAI / Anthropic)     → model inference, tools, streaming
LangChain                             → composable application pipelines
LlamaIndex                            → ingestion, retrieval, data connectors
Your policy/evaluation layer          → tenant scope, safety, release confidence
```

| Technology | Primary job | Do not mistake it for |
| --- | --- | --- |
| FastAPI | Typed Python HTTP boundary and OpenAPI contract | Authentication, durable jobs, or database tenancy by itself |
| OpenAI / Anthropic SDK | Provider-specific request/response access | Your business policy or a portable domain interface |
| LangChain | Composition, prompts, runnables, retrievers, integrations | A security boundary or a reason to skip tests |
| LlamaIndex | Data ingestion/index/query abstractions and connectors | Permission enforcement or guaranteed groundedness |

> **Success callout:** Keep provider SDK objects inside adapters. Routes,
> LangGraph nodes, evaluation code, and domain services should consume typed
> application contracts such as `ModelClient`, `Citation`, or `ToolResult`.

## 2. Provider API fundamentals: OpenAI and Anthropic

Both APIs support message-oriented generation and structured tool use, but
their object shapes, rate-limit headers, streaming events, safety settings,
token accounting, and versioning are provider-specific. Treat the following as
an application contract rather than a prompt convention:

| Contract field | Why it belongs in your platform |
| --- | --- |
| Model/version identifier | Evaluation, rollback, pricing, and incident reproduction |
| System instruction version | Policy provenance and safe change review |
| User messages | Untrusted input; redact/retain according to policy |
| Tool schema and result | Deterministic validation before action execution |
| Request/correlation ID | Traceability across API, retrieval, and tool calls |
| Usage/latency/retry outcome | Cost controls, SLOs, and provider comparison |

### Concept → code: provider-neutral gateway

```bash
source .framework-venv/bin/activate
python examples/framework_demos/provider_gateway.py
```

The default run uses a `FakeModelClient`, intentionally retries a transient
failure once, and returns a normalized structured tool call. It makes no
network request. Inspect the `OpenAIChatClient` and `AnthropicMessagesClient`
classes for real SDK request mapping; both are lazy-imported and require an
explicit `--live` argument plus a securely supplied provider key.

```python
# Colab / VS Code cell — fake mode only, no secret needed
%pip install -q -r examples/requirements-frameworks-cpu.txt
!python examples/framework_demos/provider_gateway.py
```

> **Warning:** Valid tool-call JSON is not authorization. Parse it, validate it
> against a schema, bind it to the verified tenant/subject, enforce business
> policy, and audit the outcome before calling a side-effecting service.

## 3. OpenAI versus Anthropic: interview comparison

| Dimension | Strong answer |
| --- | --- |
| API abstraction | Normalize messages, tool calls, error classes, usage, and trace fields; do not force identical provider features. |
| Structured output | Use provider-supported schemas where available, then validate again with application models. |
| Tool use | Model selects an intent; a deterministic tool gateway validates and executes it. |
| Streaming | Stream only safe partial content; preserve cancellation, back-pressure, and a final auditable outcome. |
| Reliability | Deadline + classified bounded retry + fallback/queue; never retry a side-effect blindly. |
| Data governance | Know retention, regional processing, training-use settings, and tenant consent before sending payloads. |
| Evaluation | Compare providers/models on a versioned task and safety set, sliced by tenant/use case—not vibes. |

**Interview probe:** “How would you migrate from one provider to another?”

Answer: build a provider-neutral domain interface, preserve raw provider events
only in a redacted trace, normalize capability differences explicitly, replay a
held-out evaluation set, canary by traffic slice, monitor quality/latency/cost,
and retain a fast rollback path.

## 4. LangChain: composition, not control-plane policy

LangChain is useful when composable `Runnable` chains, retrievers, output
parsers, callbacks, and provider integrations reduce repeated glue code. Use
LCEL to make data flow visible.

```text
validated request → tenant-filtered retriever → context builder → model adapter
      ↓                                                          ↓
trace + evaluation metadata ← schema validator ← structured response
```

The companion lab composes two `RunnableLambda` stages: retrieval and grounded
response shaping. The local retriever makes the important invariant visible:
the candidate set is tenant-filtered before it is ranked or cited.

## 5. LlamaIndex: data and retrieval abstractions

LlamaIndex helps organize document ingestion, node metadata, connectors,
indices, retrievers, and query engines. It is particularly useful when
knowledge-source lifecycle is a major part of the product.

| Design question | Senior answer |
| --- | --- |
| Where does tenant filtering happen? | In the retrieval/store eligibility filter, with defense in depth in the data layer. |
| How is a citation stable? | Store source ID, version, chunk position, and index snapshot; never cite only a transient rank. |
| What does re-indexing change? | Record parser/chunker/embedder/index versions; use an alias/canary for rollback. |
| When use custom code instead? | A narrow, performance-sensitive, or compliance-critical path may need explicit data access and fewer framework dependencies. |

### Concept → code: LangChain and LlamaIndex, no hosted model

```bash
python examples/framework_demos/langchain_llamaindex_rag.py
```

It uses LangChain LCEL and LlamaIndex `MockEmbedding` so the framework code is
real but the outcome remains deterministic and credential-free. The LlamaIndex
retriever receives an `ExactMatchFilter` for `tenant_id`; remove it only in a
private experiment to see why it is a security invariant, not an optimization.

## 6. FastAPI integration pattern

The existing [`agent_platform_api.py`](../examples/agent_platform_api.py)
provides typed CRUD and a deployment gate. In a production API, add the
following seams rather than turning a route into a complete agent runtime.

```text
FastAPI route → auth/tenant dependency → service command → durable operation
                                      │                     ↓
                                      └→ provider / graph / retrieval adapter
                                                               ↓
                                                          trace + audit
```

| Concern | Recommended pattern |
| --- | --- |
| Long agent run | Return an operation/run ID; use a durable worker or workflow; expose status/streaming separately. |
| Idempotency | Require client key for create/deploy commands; persist request fingerprint and terminal result. |
| Authentication | Verify OIDC/JWT at the edge; derive tenant/scopes rather than accepting them in JSON. |
| Validation | Pydantic request/response models plus domain policy validation. |
| Testing | `TestClient` contract tests, fake model/retriever tests, integration tests for adapters, evaluation regression gate. |

## 7. Use-case drill: citation-aware support API

**Prompt:** “Build an endpoint that answers a customer’s deployment-policy
question using company documents.”

1. Verify identity and derive tenant/ACL context.
2. Use LangChain or custom composition to invoke a tenant-filtered retriever.
3. Use LlamaIndex or a purpose-built index to return document/version metadata.
4. Call an OpenAI or Anthropic adapter through a model gateway, with a bounded
   context and structured `answer + citations` schema.
5. Validate citations; refuse or return an evidence-missing response when no
   authorized source supports the claim.
6. Record safe trace fields, measure groundedness/retrieval recall/latency, and
   evaluate the change before release.

Proceed to [Module 09: Agent Frameworks and MCP](09_agent_frameworks_and_mcp.md).
