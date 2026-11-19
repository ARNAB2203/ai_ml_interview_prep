# Module 01 — Foundations: Production Python, ML, Tokens, and BPE

**Level:** Beginner → Intermediate  
**Interview goal:** Explain a modeling or tokenization choice in terms of
correctness, cost, failure behavior, and operational impact.

## 1. Production Python is an ML feature

In an agent platform, Python is not merely glue around a model. It defines API
contracts, validates configuration, scopes tenancy, records trace context, and
makes failures diagnosable. A high-quality answer therefore mentions:

| Concern | Senior-level practice |
| --- | --- |
| Boundaries | Keep provider, retrieval, tools, policy, and HTTP adapters behind interfaces. |
| Types | Use `dataclass` or Pydantic models at ingress/egress; reject invalid state early. |
| Errors | Make failures typed, actionable, and safe to expose; never leak tenant data or secrets. |
| Tests | Unit-test deterministic policy and state transitions; integration-test adapter contracts. |
| Packaging | Pin compatible dependencies, build immutable artifacts, and record version metadata. |

> **Interview probe:** “Why not put model calls directly inside a FastAPI route?”
>
> A strong answer: it couples transport, model vendor, retry policy, tracing,
> and business policy. A service layer with a provider interface lets us test
> deterministic behavior, apply uniform budgets, and change providers without
> changing the public API.

## 2. ML evaluation mental model

An ML or GenAI feature needs a measurable objective before implementation.

| Term | Meaning | Agent-platform example |
| --- | --- | --- |
| Offline metric | Repeatable score on a versioned dataset | Citation precision on a fixed RAG evaluation set |
| Online metric | Behavior measured in production | Task completion, latency, tool-error rate, cost per run |
| Slice | Meaningful subgroup | Tenant, document type, language, tool, or prompt version |
| Guardrail | Hard constraint that blocks unsafe behavior | No tool call without authorization or schema validation |
| Regression gate | Threshold needed before release | Grounded-answer rate must not fall below the baseline |

Accuracy alone is insufficient. A system can be accurate on average yet unsafe
for one tenant, unaffordable at the 95th percentile, or impossible to debug.

## 3. Tokenization: text becomes model input

Most language models consume IDs, not raw strings. A tokenizer maps text to
tokens, maps tokens to IDs, and reverses the map for output. The reference
notebooks correctly start with regex tokenization before introducing a production
library; that sequence exposes the trade-offs.

```text
raw text → token pieces → vocabulary lookup → integer IDs → model
                    ↑                              ↓
                 decode ← inverse vocabulary ← generated IDs
```

### Vocabulary and unknown tokens

A word-level vocabulary is simple but has an out-of-vocabulary (OOV) problem.
Mapping every unseen word to `<unk>` loses distinctions: `pneumonoultramic...`
and `tenant_42` become identical. A subword algorithm such as byte-pair encoding
(BPE) learns frequent pieces, allowing novel words to be represented as known
parts.

| Choice | Advantage | Cost / risk |
| --- | --- | --- |
| Word tokenizer | Easy to inspect | OOV explosion; large vocabulary |
| Character tokenizer | No OOV | Long sequences and weak semantic units |
| BPE / subword tokenizer | Good OOV behavior and vocabulary efficiency | Merge rules must match training/runtime exactly |
| Vendor tokenizer | Correct alignment to a hosted model | Version change can alter counts, limits, and cost |

> **Production warning:** Token counts influence context limits, latency, and
> pricing. Record tokenizer/model versions with traces and evaluation results.

## 4. Concept → code: build a deterministic tokenizer and BPE trainer

The companion script follows the reference notebook style: toy text first,
explicit prints, an initial vocabulary, an `<unk>` boundary, then BPE merges.

```bash
python examples/tokenizer_bpe.py
```

Expected observations:

1. Punctuation is separated from words by regex tokenization.
2. An unseen whole word becomes `<unk>` in a fixed word vocabulary.
3. BPE learns frequent character pairs and can encode an unseen related word as
   subword pieces rather than a single unknown token.

Try changing the training corpus to include domain phrases such as
`agent_manifest` and `tenant_isolation`. Explain whether a general-purpose
tokenizer should be retrained for those terms; usually it should not unless the
model is trained or adapted with the same tokenizer.

## 5. Embeddings and similarity—what to say in an interview

An embedding is a vector representation optimized so related content is nearby
under a similarity function, commonly cosine similarity.

```text
cosine(a, b) = dot(a, b) / (||a|| × ||b||)
```

Use the word “representation” rather than “truth.” Embeddings are lossy and can
miss negation, identifiers, recency, or tenant policy. Retrieval should filter
by tenant and authorization *before* semantic ranking, then attach source
metadata for audit and citation.

## 6. Mini design exercise

**Prompt:** “A user uploads a custom agent package that contains prompt
templates. How should you detect its token budget before deployment?”

Model answer outline:

1. Parse and validate the manifest before reading templates.
2. Choose the exact tokenizer associated with the selected runtime model.
3. Count system prompt, tool schemas, retrieval allowance, conversation reserve,
   and output reserve separately.
4. Reject or require explicit truncation when the deterministic worst case
   exceeds the model context window.
5. Persist the calculation and tokenizer/runtime version in deployment metadata.

## 7. Senior follow-up questions

1. How do you test that a tokenizer upgrade did not silently change context
   budgeting?
2. When is a small local embedding model safer than a hosted embedding API?
3. Why can an evaluation score improve while customer experience worsens?
4. What is the difference between a validation set, a holdout set, and an online
   canary?

Proceed to [Module 02: Agent Design and Orchestration](02_agent_design_and_orchestration.md).
