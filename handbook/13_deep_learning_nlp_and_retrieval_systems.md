# Module 13 — Deep Learning, NLP, Transformers, and Retrieval Systems

**Level:** Intermediate → Advanced
**Interview goal:** Explain architecture and ranking behavior from tensors and
data to user-visible quality, rather than treating embeddings or transformers
as black boxes.

## 1. Neural-network fundamentals

```text
input → affine layer → activation → … → logits → loss
  ↑                                         ↓
  └────────── gradients / optimizer update ─┘
```

| Topic | What to articulate |
| --- | --- |
| Forward pass | Computes predictions from current weights and inputs |
| Loss | Translates task objective into an optimizable scalar; must match output/label semantics |
| Backpropagation | Chain rule computes gradients of loss with respect to parameters |
| Optimizer | Uses gradients plus state (for example momentum/Adam) to update parameters |
| Initialization/normalization | Keeps activations and gradients in a trainable range |
| Dropout/regularization | Limits overfit but changes train/eval behavior |
| Gradient check | Compare autodiff with a finite-difference approximation for custom operations |

### Debug sequence when training does not learn

1. Overfit a tiny, known-good batch.
2. Verify feature/label alignment, range, dtype, masks, and output activation.
3. Inspect loss before/after one update, gradient norms, parameter changes, and
   train/eval mode.
4. Reduce complexity: fixed seed, no augmentation, smaller learning rate, known
   optimizer, single worker/device.
5. Reintroduce data pipeline, regularization, precision, and distribution in
   controlled steps.

## 2. Architecture selection

| Architecture | Strength | Limitation / interview trade-off |
| --- | --- | --- |
| CNN | Local/spatial patterns, weight sharing | Receptive field and spatial inductive bias may not fit arbitrary sequences |
| RNN | Ordered stateful sequence processing | Long-range gradient/parallelism limitations |
| LSTM | Gates/cell state reduce vanishing-gradient pressure | More parameters/latency than simpler recurrent options |
| GRU | Compact gated recurrent alternative | Fewer controls than LSTM; benchmark task-specific quality |
| Transformer | Parallel attention and flexible context | Attention/context/KV-cache cost grows rapidly with sequence length |
| Frozen pretrained encoder + head | Low-data adaptation | Domain shift and missed task-specific behavior |
| Selective fine-tune / LoRA | Higher adaptation capacity | Forgetting, evaluation burden, serving/version complexity |

### Concept → code: gradients, shapes, attention, and a TF-IDF baseline

```bash
python examples/ml_ds_demos/deep_learning_nlp_foundations.py
```

The local lab performs a finite-difference gradient check, creates CNN/RNN/LSTM/
GRU modules to inspect exact shapes, applies a causal attention mask, and uses a
TF-IDF n-gram baseline. A strong candidate can say why each baseline exists
before proposing a larger transformer.

## 3. Classical NLP remains useful

| Technique | Good use | Why it can beat an LLM/embedding approach |
| --- | --- | --- |
| Normalization/tokenization | Controlled text preprocessing | Cheap, deterministic, auditable |
| N-grams/TF-IDF | Exact technical terms, sparse labels, fast baseline | Strong lexical signal with little data/latency |
| NER / structured extraction | Known entities and downstream schema | Deterministic rules/models can be easier to validate |
| Static embeddings | Lightweight similarity | Lower cost for stable vocabulary |
| Contextual embeddings | Semantic retrieval/classification | Better context sensitivity; version/cost/latency burden |

Use error analysis to decide. A specific product identifier, policy number, or
negated phrase may require exact lexical retrieval even when semantic similarity
is available.

## 4. Transformer and attention essentials

For scaled dot-product attention:

```text
Attention(Q, K, V) = softmax((QKᵀ / √d_k) + mask) V
```

| Element | Engineering implication |
| --- | --- |
| Query/key/value | Content-based selection of context; tensor shapes must align |
| Scale `√d_k` | Keeps logits numerically manageable as width grows |
| Causal/padding masks | Enforce no future-token access and ignore padding safely |
| Multi-head attention | Learns multiple relation subspaces; costs more memory/compute |
| Positional information | Provides order absent from a set-based attention operation |
| KV cache | Reduces repeat work during generation but consumes memory per token/request |
| Long context | Increases retrieval/prompt cost and can dilute relevant evidence |

When context cost is a bottleneck, first measure. Consider chunking, retrieval,
reranking, summarization with provenance, smaller context, cache policy, sparse/
windowed attention, or a model/runtime designed for the workload.

## 5. Vector databases and hybrid retrieval

The existing RAG modules cover grounding and tenant filtering. A production
retrieval system also needs index operations and ranking evaluation.

| Component | Decision / trade-off |
| --- | --- |
| Exact k-NN | Best recall, expensive at scale |
| HNSW | High recall/low latency graph search, larger memory and tuning/deletion complexity |
| IVF/PQ | Lower memory/latency with coarse partition/compression, recall trade-off |
| Metadata/ACL filter | Security invariant; selectivity affects index/query design |
| Sparse search | Exact term/identifier strength |
| Dense search | Semantic paraphrase strength |
| Hybrid + RRF | Fuses complementary rankings; tune/evaluate rather than assume benefit |
| Reranker | Improves top-k relevance, adds latency/cost |
| Update/delete | Source/version lineage, tenant offboarding, eventual-consistency and rollback plan |

### Ranking metrics

| Metric | Meaning |
| --- | --- |
| Recall@k | Did any expected evidence appear in top-k? |
| MRR | How early did the first expected document appear? |
| NDCG | Does ranking reward relevant documents near the top, including graded relevance? |
| Context precision | How much retrieved context actually supports the answer? |
| Groundedness | Are response claims supported by cited context? |

### Concept → code: hybrid retrieval and index lifecycle

```bash
python examples/ml_ds_demos/hybrid_retrieval_operations.py
```

The demo applies a tenant pre-filter, lexical and hashed-dense ranking, reciprocal
rank fusion, Recall@k/MRR/NDCG, a document deletion check, Little’s Law capacity
math, and an approximate KV-cache calculation. It is deliberately an exact,
small teaching index—not a substitute for a managed vector database.

## 6. High-value incident drill

**Prompt:** “After an embedding migration, answers are less relevant only for
one tenant and deleted documents sometimes appear.”

Answer sequence:

1. Freeze/index alias rollback if safety or privacy is at risk.
2. Compare tenant-specific source snapshots, embedding dimensions/model
versions, chunker, filters, cache keys, delete propagation, and query traces.
3. Re-run fixed relevance cases with Recall@k/MRR/NDCG and inspect ranking
changes—not only final answer fluency.
4. Confirm deletion behavior in the vector store, cache, replicas, backups, and
downstream citations.
5. Reindex incrementally behind an alias, canary by tenant, measure p95/cost,
and retain a rollbackable prior index.

Proceed to [Module 14: MLOps, Serving, and Responsible AI](14_mlops_serving_and_responsible_ai.md).
