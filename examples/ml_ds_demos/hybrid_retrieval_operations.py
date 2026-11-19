"""Hybrid retrieval, ranking metrics, index lifecycle, and capacity math.

Run locally (standard library only):
    python examples/ml_ds_demos/hybrid_retrieval_operations.py

The index is deliberately small and exact. It teaches the contracts behind a
production vector store: tenant filtering before ranking, sparse+dense fusion,
evaluation, fresh versions, deletion, and capacity budgets. It is not an HNSW
or IVF implementation.
"""

# %% Imports
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
from math import log, sqrt
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class IndexedDocument:
    document_id: str
    tenant_id: str
    version: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    document: IndexedDocument
    sparse_rank: int
    dense_rank: int
    fused_score: float


@dataclass(frozen=True)
class EvaluationCase:
    tenant_id: str
    query: str
    relevant_document_ids: frozenset[str]


class HybridIndex:
    """Small exact index with lexical+dense rank fusion and explicit tenancy."""

    def __init__(self, dimensions: int = 48) -> None:
        self.dimensions = dimensions
        self._documents: dict[str, IndexedDocument] = {}

    def upsert(self, document: IndexedDocument) -> None:
        self._documents[document.document_id] = document

    def delete(self, document_id: str) -> bool:
        return self._documents.pop(document_id, None) is not None

    def _embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimensions] += 1.0 if value & 1 else -1.0
        norm = sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector) if norm else tuple(vector)

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    def _sparse_scores(self, documents: list[IndexedDocument], query: str) -> dict[str, float]:
        query_counts = Counter(tokenize(query))
        document_frequency = Counter(token for document in documents for token in set(tokenize(document.text)))
        scores: dict[str, float] = {}
        for document in documents:
            document_counts = Counter(tokenize(document.text))
            scores[document.document_id] = sum(
                query_count * document_counts[token] * (log((len(documents) + 1) / (document_frequency[token] + 1)) + 1)
                for token, query_count in query_counts.items()
            )
        return scores

    @staticmethod
    def _ranks(scores: dict[str, float]) -> dict[str, int]:
        return {
            document_id: rank
            for rank, (document_id, _) in enumerate(sorted(scores.items(), key=lambda item: (-item[1], item[0])), start=1)
        }

    def search(self, tenant_id: str, query: str, limit: int = 3, rrf_k: int = 60) -> list[SearchResult]:
        # Security invariant: only eligible tenant documents enter either ranking.
        eligible = [document for document in self._documents.values() if document.tenant_id == tenant_id]
        sparse_scores = self._sparse_scores(eligible, query)
        query_embedding = self._embed(query)
        dense_scores = {document.document_id: self._cosine(query_embedding, self._embed(document.text)) for document in eligible}
        sparse_ranks, dense_ranks = self._ranks(sparse_scores), self._ranks(dense_scores)
        fused = {
            document.document_id: 1 / (rrf_k + sparse_ranks[document.document_id])
            + 1 / (rrf_k + dense_ranks[document.document_id])
            for document in eligible
        }
        by_id = {document.document_id: document for document in eligible}
        return [
            SearchResult(by_id[document_id], sparse_ranks[document_id], dense_ranks[document_id], score)
            for document_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]


def evaluate_retrieval(index: HybridIndex, cases: list[EvaluationCase], limit: int = 3) -> dict[str, float]:
    """Compute recall@k, MRR, and binary-relevance NDCG for a fixed evaluation set."""
    recall, reciprocal_rank, ndcg = [], [], []
    for case in cases:
        result_ids = [result.document.document_id for result in index.search(case.tenant_id, case.query, limit)]
        relevant_positions = [position for position, document_id in enumerate(result_ids, start=1) if document_id in case.relevant_document_ids]
        recall.append(bool(relevant_positions))
        reciprocal_rank.append(1 / relevant_positions[0] if relevant_positions else 0.0)
        dcg = sum(1 / log(position + 1, 2) for position in relevant_positions)
        ideal_count = min(len(case.relevant_document_ids), limit)
        ideal_dcg = sum(1 / log(position + 1, 2) for position in range(1, ideal_count + 1))
        ndcg.append(dcg / ideal_dcg if ideal_dcg else 0.0)
    return {
        "recall_at_k": sum(recall) / len(cases),
        "mrr": sum(reciprocal_rank) / len(cases),
        "ndcg": sum(ndcg) / len(cases),
    }


def little_law_concurrency(arrival_rate_per_second: float, p95_seconds: float) -> float:
    """Approximate required in-flight capacity: L = λ × W."""
    if arrival_rate_per_second < 0 or p95_seconds < 0:
        raise ValueError("arrival rate and duration must be non-negative")
    return arrival_rate_per_second * p95_seconds


def approximate_kv_cache_gib(
    layers: int, hidden_size: int, tokens_per_request: int, concurrent_requests: int, dtype_bytes: int = 2
) -> float:
    """Approximate K+V cache bytes; real architectures may use grouped-query attention."""
    bytes_required = 2 * layers * hidden_size * tokens_per_request * concurrent_requests * dtype_bytes
    return bytes_required / 2**30


def build_demo_index() -> HybridIndex:
    index = HybridIndex()
    index.upsert(
        IndexedDocument(
            "acme-canary-v1", "acme", "1", "Deployment rollback uses an immutable artifact and canary rollback plan."
        )
    )
    index.upsert(
        IndexedDocument(
            "acme-index-v2", "acme", "2", "Vector index deletion removes stale tenant documents before retrieval."
        )
    )
    index.upsert(
        IndexedDocument(
            "globex-approval-v1", "globex", "1", "Regulated production deployment requires human approval before release."
        )
    )
    return index


def demo() -> None:
    index = build_demo_index()
    cases = [
        EvaluationCase("acme", "How do we rollback a canary deployment?", frozenset({"acme-canary-v1"})),
        EvaluationCase("acme", "How are stale vector documents deleted?", frozenset({"acme-index-v2"})),
    ]
    metrics = evaluate_retrieval(index, cases)
    acme_results = index.search("acme", "regulated human approval deployment")
    deleted = index.delete("acme-canary-v1")
    after_delete = index.search("acme", "rollback canary deployment")
    concurrency = little_law_concurrency(arrival_rate_per_second=20, p95_seconds=1.2)
    kv_cache = approximate_kv_cache_gib(32, 4096, 4096, int(concurrency))

    print("Hybrid retrieval metrics:", metrics)
    print("Acme-only results for a Globex-looking query:", [result.document.document_id for result in acme_results])
    print("Deletion state:", {"deleted": deleted, "remaining_results": [r.document.document_id for r in after_delete]})
    print("Capacity estimates:", {"concurrent_runs": concurrency, "approx_kv_cache_gib": kv_cache})

    assert metrics["recall_at_k"] == 1.0
    assert all(result.document.tenant_id == "acme" for result in acme_results)
    assert deleted and "acme-canary-v1" not in {result.document.document_id for result in after_delete}
    assert concurrency == 24.0
    print("PASS: hybrid ranking, isolation, index lifecycle, and capacity checks completed.")


if __name__ == "__main__":
    demo()
