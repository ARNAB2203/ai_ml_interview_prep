# Module 12 — Data Engineering, Mathematics, and Classical ML Foundations

**Level:** Beginner → Advanced
**Interview goal:** Demonstrate that an AI/GenAI engineer can reason about the
data and statistical foundation beneath an LLM workflow—not only call a model
API.

## 1. Evidence-driven engineering loop

Use the same disciplined loop for a classifier, retriever, agent, or model API:

```text
Learn assumptions → build a small baseline → break it with edge cases
          ↓                    ↓                         ↓
measure/debug evidence ← deploy a versioned, observable solution
```

At senior level, name the assumption, the failure mode, the metric, and the
rollback path. A framework is never a substitute for this loop.

## 2. Production Python and data-contract basics

| Topic | What to know | Production use case |
| --- | --- | --- |
| Types/dataclasses/Pydantic | Explicit inputs and invariants | Ingestion/API boundaries that reject malformed records |
| Iterators/generators | Lazy processing and bounded memory | Stream large event/document collections |
| Context managers | Deterministic resource cleanup | Files, connections, trace spans, temporary artifacts |
| Async versus threads/processes | I/O concurrency versus CPU/GIL work | Concurrent tool calls versus CPU-heavy parsing/training |
| Logging/profiling | Structured evidence, not print debugging | Diagnose a memory-growing service or slow ETL stage |
| Unit/property tests | Assert behavior and invariants | Schema, parsing, policy, and metric regression tests |

### Data contracts before features

Validate required fields, types, uniqueness, allowed ranges, nullability,
schema version, lineage, and deletion/retention policy before model code sees a
row. Record a data manifest containing source snapshot, extraction time, schema
version, feature code version, and row counts.

> **Warning:** Missing values are not always invalid values. Decide whether a
> null means “not collected,” “not applicable,” “delayed,” or “data failure”;
> preserve that distinction as a feature/quality signal rather than blindly
> imputing it away.

## 3. NumPy, Pandas, and algorithmic building blocks

| Concept | Interview signal | AI/platform application |
| --- | --- | --- |
| Array shape/dtype/broadcasting | Can explain vectorized operations and numerical memory | Embedding batches, attention tensors, metric arrays |
| Copy versus view | Avoids accidental mutation/memory blowups | Feature transforms and sampled trace frames |
| Joins/groupby/windows | Understands cardinality and aggregation | Tenant metrics, event features, cost/latency analysis |
| Hash maps | Average O(1) lookup with caveats | Deduplication, request cache, idempotency record |
| Heap / top-k | O(n log k), not full sort | Candidate retrieval and rate-limited job priority |
| BFS/DFS | Traversal and cycle detection | Dependency graph, tool/workflow graph, lineage inspection |
| LRU cache | Memory-bounded recency policy | Tenant-aware embedding or retrieval cache |

**Scenario:** “A join doubles your events.” First inspect join keys, uniqueness
on both sides, null-key behavior, expected cardinality, and pre/post row counts.
Do not start by dropping duplicates—doing so can hide a data-contract breach.

## 4. Mathematical cheat sheet with decisions

| Idea | Compact form | What decision it changes |
| --- | --- | --- |
| Cosine similarity | `a·b / (||a|| ||b||)` | Embedding normalization/comparability and semantic ranking |
| Logistic probability | `sigmoid(z) = 1 / (1 + e^-z)` | Thresholded risk score versus calibrated probability |
| Binary cross-entropy | `-[y log p + (1-y) log(1-p)]` | Classification objective and class-imbalance weighting |
| Gradient descent | `θ ← θ - η∇L(θ)` | Learning-rate/optimization debugging when loss diverges |
| L1 / L2 | `λΣ|w|` / `λΣw²` | Sparsity versus smooth shrinkage and multicollinearity handling |
| Precision / recall | `TP/(TP+FP)` / `TP/(TP+FN)` | False-alert versus missed-incident operating point |
| Confidence interval | estimate ± uncertainty | Whether a small experiment lift is likely actionable |

### Probability, statistics, and experiments

Use a metric only after defining the population, label availability/latency,
sampling, slice, error cost, and decision. Accuracy is often misleading on
imbalanced data; PR-AUC, recall at a fixed precision, calibration/Brier score,
and segment metrics may better match the product risk.

| Question | Strong answer |
| --- | --- |
| Probability versus likelihood? | Probability describes outcomes given parameters; likelihood compares parameter values given observed outcomes. |
| Interpret a p-value? | It is not the probability a hypothesis is true; it is a result extremeness measure under the stated null/model assumptions. |
| Type I versus II error? | False positive versus false negative; cost and threshold determine which is worse. |
| Accuracy rose 1%—ship? | Check confidence/uncertainty, slices, calibration, cost, safety, label quality, and pre-registered decision threshold. |

## 5. Classical ML: baseline before sophistication

| Topic | When it is valuable | Key failure mode |
| --- | --- | --- |
| Linear/logistic regression | Interpretable baseline, calibrated risk candidate | Leakage, unscaled features, threshold chosen by habit |
| Decision tree/random forest | Nonlinear structured data and feature interactions | Deep-tree overfit, unstable importance interpretation |
| Gradient boosting/XGBoost | Strong tabular baseline | Validation tuning leakage and uncalibrated probabilities |
| K-means | Compact roughly spherical clusters | Scale sensitivity and arbitrary cluster meaning |
| DBSCAN | Density clusters/noise | Parameter sensitivity and varying-density data |
| PCA/SVD | Compression/visualization/denoising | Treating components as causal explanations |

### Leakage-safe evaluation

Random row splits fail when the same customer, document lineage, device, or
future information leaks across partitions. Use group/time/dependency-aware
splits and fit imputation/scaling/encoding on the training partition only.

```text
raw events → data contract → group/time split
     → fit preprocessing on train only → train/tune
     → frozen test + slice metrics → versioned promotion decision
```

## 6. Concept → code: data contract to model comparison

```bash
source .framework-venv/bin/activate
python -m pip install -r examples/requirements-ml-ds.txt
python examples/ml_ds_demos/data_classical_ml.py
```

The lab creates synthetic grouped events, allows documented missing values,
enforces a data contract, prevents customer leakage with `GroupShuffleSplit`,
fits logistic regression/random forest/XGBoost baselines, chooses a threshold
to meet a recall floor, evaluates calibration, clusters numeric events, and
shows cosine similarity plus gradient descent.

> **Expected observation:** Model scores differ, but none is declared “best”
> without a business objective, calibration/slice check, operational cost, and
> versioned evaluation evidence.

## 7. Senior interview drills

1. A random split scores well, but future-time evaluation fails. What data or
   deployment assumption has likely been violated?
2. One feature dominates a model. How do you check target leakage, proxy bias,
   stale data, and a legitimate causal signal?
3. Recall must remain above 90% while false alerts are expensive. How do you
   select and monitor a threshold?
4. A model score improves but calibration worsens. Which downstream decision is
   at risk and how do you repair it?

Proceed to [Module 13: Deep Learning, NLP, and Retrieval Systems](13_deep_learning_nlp_and_retrieval_systems.md).
