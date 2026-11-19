# AI/ML Interview Preparation Handbook

## Senior Agentic Platform Engineer (Python/ML)

| Document property | Value |
| --- | --- |
| Intended audience | Senior software/AI engineer: up to 12 years total experience; 3+ years AI/ML or GenAI |
| Target role | Senior Agentic Platform Engineer (Python/ML) |
| Primary emphasis | Production Python, agent platforms, RAG, multi-tenancy, quality, and operability |
| Learning mode | Concept → runnable code → expected observation → interview probe → production trade-off |
| Companion code | [`examples/`](../examples/) |

> **Goal** — Be able to turn a vague request such as “build an agent platform”
> into a secure, testable, observable design with explicit trade-offs. A senior
> candidate should be able to defend the choices, not merely draw the boxes.

## 1. Role decomposition

The supplied JD describes a multi-tenant SaaS platform that creates,
configures, tests, deploys, and manages AI agents and their toolkits. The
interview is therefore likely to probe the following capabilities together.

| JD signal | What an interviewer wants to hear | Where to practice |
| --- | --- | --- |
| Production Python services | Types, validation, tests, packages, failure handling, and debuggability | 01, 04, 05 |
| Agent API and lifecycle | Clear resource model, idempotency, state transitions, API contracts | 02, 04, 06 |
| Tool-using agents | Structured tool schemas, authorization, retries, bounded autonomy | 02 |
| RAG / KB integration | Retrieval quality, citations, freshness, access filtering, evaluation | 03 |
| Multi-tenant platform | Tenant scoping at every layer, data isolation, quotas, auditability | 04, 05, 06 |
| Packaging and A2A/MCP-style integration | Manifest validation, compatibility, sandboxing, versioning | 04, 05 |
| Evaluation practices | Offline test sets, gates, regressions, human review and telemetry | 03, 05 |
| Docker/AWS runtime patterns | Immutable artifacts, health checks, secrets, rollout, observability | 05, 06 |
| Framework/API fluency | Provider adapters, LangChain, LlamaIndex, FastAPI integration | 08 |
| Agent orchestration | LangGraph, MCP, AutoGen, CrewAI, Google ADK decision patterns | 09 |
| Model engineering | PyTorch, JAX, Hugging Face, PEFT/LoRA/QLoRA, DeepSpeed | 10, 11 |

## 2. How to use this handbook

The supplied reference notebooks teach in small, visible steps: explain a
single concept, run a small cell, inspect output, then add one production
concern. This handbook uses the same rhythm.

1. Read the **mental model** and say it back in your own words.
2. Run the linked script or `# %%` cell.
3. Change one input deliberately and explain the resulting behavior.
4. Answer the **interview probe** in two minutes.
5. Finish with the **production trade-off**; this is what differentiates a
   senior answer from a tutorial answer.

> **Warning — deterministic defaults are intentional.** The default examples
> use local, rule-based models, mock embeddings, or tiny local models so a
> reviewer can reproduce outputs without secrets, a GPU, or a network
> connection. Clearly labelled live-provider and accelerator paths are opt-in.
> In production, replace only the adapter boundary—not authorization,
> evaluation, or observability controls—with a managed model or vector service.

## 3. Six-week progression: Beginner to Master

| Week | Level | Outcomes | Evidence you should produce |
| --- | --- | --- | --- |
| 1 | Beginner | Explain tokenization, embeddings, supervised evaluation, Python service fundamentals | Run `tokenizer_bpe.py`; explain `<unk>` versus subwords |
| 2 | Intermediate | Implement structured tool calls and bounded workflow state | Extend `tool_using_agent.py` with one safe tool |
| 3 | Intermediate | Build RAG retrieval, citations, and an offline evaluation set | Change a document and observe the evaluation failure |
| 4 | Advanced | Design a tenant-aware Agent API and lifecycle state model | Run API tests; defend tenant boundaries |
| 5 | Advanced | Validate packages and reason about protocol/version compatibility | Reject a malicious archive path in the validator |
| 6 | Master | Lead a system-design interview across quality, security, scale, and operations | Whiteboard the platform from Module 06; run a mock loop |

## 4. Readiness rubric

Score each dimension from 0 to 3 after completing the modules.

| Score | Meaning |
| --- | --- |
| 0 | Can define terms but cannot implement or diagnose them. |
| 1 | Can implement a happy path with guidance. |
| 2 | Can explain failure modes and make a justified trade-off. |
| 3 | Can design the boundary, operational controls, tests, and rollout plan. |

| Dimension | Target score before a senior interview |
| --- | --- |
| Python/API correctness | 3 |
| Agent/tool safety | 3 |
| RAG and evaluation | 3 |
| Multi-tenancy and security | 3 |
| Deployment and observability | 2–3 |
| Communication and system design | 3 |

## 5. Fast interview answer framework

Use this sequence when an interviewer asks a broad design question.

```text
Clarify outcome and non-functional constraints
        ↓
Name boundaries: tenant, identity, data, model, tool, deployment
        ↓
Propose the smallest end-to-end path and its contracts
        ↓
Describe failure, security, and evaluation controls
        ↓
State metrics, rollout plan, and the trade-off you would revisit
```

For example, do not answer “I would use a vector database.” Say which data is
tenant-filtered, how a result is cited, how recall is measured, what happens
when retrieval is empty, and how the change is released safely.

## 6. Companion code quick reference

| Script | Demonstrates | Run command |
| --- | --- | --- |
| `tokenizer_bpe.py` | Regex tokenization, vocabulary IDs, `<unk>`, deterministic BPE | `python examples/tokenizer_bpe.py` |
| `tool_using_agent.py` | Structured plan, authorized tools, retries, audit events | `python examples/tool_using_agent.py` |
| `rag_evaluation.py` | Retrieval, citations, offline quality gates | `python examples/rag_evaluation.py` |
| `agent_platform_api.py` | FastAPI CRUD, schemas, deployment transition, tenant isolation | `python examples/agent_platform_api.py` |
| `manifest_validator.py` | ZIP manifest safety and compatibility checks | `python examples/manifest_validator.py` |
| `framework_demos/provider_gateway.py` | OpenAI/Anthropic adapter boundary, fake provider, structured tool calls | `python examples/framework_demos/provider_gateway.py` |
| `framework_demos/langchain_llamaindex_rag.py` | LCEL composition and LlamaIndex tenant metadata filters | `python examples/framework_demos/langchain_llamaindex_rag.py` |
| `framework_demos/langgraph_approval_graph.py` | Stateful approval routing | `python examples/framework_demos/langgraph_approval_graph.py` |
| `framework_demos/mcp_knowledge_server.py` | Real local FastMCP discovery and narrow tools | `python examples/framework_demos/mcp_knowledge_server.py` |
| `framework_demos/pytorch_hf_peft.py` | CPU PyTorch, local Hugging Face model, and LoRA step | `python examples/framework_demos/pytorch_hf_peft.py` |
| `framework_demos/jax_training_patterns.py` | JAX PRNG, `grad`, `jit`, and `vmap` | `python examples/framework_demos/jax_training_patterns.py` |
| `framework_demos/deepspeed_zero_config.py` | ZeRO configuration validation and memory planning | `python examples/framework_demos/deepspeed_zero_config.py` |

The first seven modules are the core JD-aligned track. Modules 08–11 are the
stack-extension track requested for framework and model-engineering interviews.
Continue with [Module 01: Foundations, Python, and ML](01_foundations_python_and_ml.md).
