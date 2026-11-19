"""Tenant-aware RAG and a small offline evaluation gate.

Run locally:
    python examples/rag_evaluation.py

Google Colab setup cell:
    # This lesson uses only the Python standard library.

The hashing embedder is deterministic teaching code. In production, replace it
with a versioned embedding service while preserving tenant filters, citations,
evaluation datasets, and release thresholds.
"""

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import sqrt
from time import perf_counter
import re


WORD_PATTERN = re.compile(r"[a-z0-9_]+")


# %% Step 1 — deterministic local embeddings
class HashingEmbedder:
    """Map tokens into a stable, normalized vector without external packages."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for token in WORD_PATTERN.findall(text.lower()):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        length = sqrt(sum(value * value for value in vector))
        return tuple(value / length for value in vector) if length else tuple(vector)


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


# %% Step 2 — data carries tenant and source identity
@dataclass(frozen=True)
class Document:
    document_id: str
    tenant_id: str
    text: str
    version: str


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float


class TenantKnowledgeBase:
    """In-memory index that filters by tenant before similarity ranking."""

    def __init__(self, embedder: HashingEmbedder) -> None:
        self._embedder = embedder
        self._records: list[tuple[Document, tuple[float, ...]]] = []

    def add(self, document: Document) -> None:
        self._records.append((document, self._embedder.embed(document.text)))

    def retrieve(self, tenant_id: str, query: str, limit: int = 2) -> list[RetrievedDocument]:
        query_vector = self._embedder.embed(query)
        # The security boundary: eligibility is tenant-filtered before top-k.
        eligible = [
            RetrievedDocument(document, cosine(query_vector, vector))
            for document, vector in self._records
            if document.tenant_id == tenant_id
        ]
        return sorted(eligible, key=lambda item: (-item.score, item.document.document_id))[:limit]


# %% Step 3 — response includes source citations instead of hidden context
@dataclass(frozen=True)
class RagAnswer:
    text: str
    citations: tuple[str, ...]
    latency_ms: float


class GroundedResponder:
    """Small response adapter that makes the grounding contract observable."""

    def __init__(self, knowledge_base: TenantKnowledgeBase) -> None:
        self._knowledge_base = knowledge_base

    def answer(self, tenant_id: str, question: str) -> RagAnswer:
        started = perf_counter()
        retrieved = self._knowledge_base.retrieve(tenant_id, question)
        if not retrieved or retrieved[0].score <= 0:
            return RagAnswer(
                "I do not have authorized evidence to answer that question.",
                (),
                (perf_counter() - started) * 1_000,
            )

        top = retrieved[0].document
        response = f"According to {top.document_id} (version {top.version}): {top.text}"
        return RagAnswer(response, (top.document_id,), (perf_counter() - started) * 1_000)


# %% Step 4 — versioned offline evaluation cases and quality gate
@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    tenant_id: str
    question: str
    expected_citations: frozenset[str]
    max_latency_ms: float = 250.0


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    retrieval_hit: bool
    grounded: bool
    cross_tenant_leak: bool
    latency_ms: float


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[CaseResult, ...]
    retrieval_recall: float
    grounded_rate: float
    cross_tenant_leaks: int
    p95_like_latency_ms: float
    passed: bool


def evaluate(responder: GroundedResponder, knowledge_base: TenantKnowledgeBase, cases: list[EvaluationCase]) -> EvaluationReport:
    """Evaluate citations, grounding, isolation, and a simple latency threshold."""
    results: list[CaseResult] = []
    for case in cases:
        answer = responder.answer(case.tenant_id, case.question)
        retrieval_hit = case.expected_citations.issubset(set(answer.citations))
        cited_docs = [
            document
            for document, _ in knowledge_base._records  # Teaching-only inspection; use an audited store API in production.
            if document.document_id in answer.citations
        ]
        cross_tenant_leak = any(document.tenant_id != case.tenant_id for document in cited_docs)
        grounded = bool(answer.citations) and all(citation in answer.text for citation in answer.citations)
        passed = retrieval_hit and grounded and not cross_tenant_leak and answer.latency_ms <= case.max_latency_ms
        results.append(
            CaseResult(
                case.case_id,
                passed,
                retrieval_hit,
                grounded,
                cross_tenant_leak,
                answer.latency_ms,
            )
        )

    count = len(results)
    retrieval_recall = sum(result.retrieval_hit for result in results) / count
    grounded_rate = sum(result.grounded for result in results) / count
    cross_tenant_leaks = sum(result.cross_tenant_leak for result in results)
    p95_like_latency_ms = max(result.latency_ms for result in results)
    passed = (
        retrieval_recall >= 0.95
        and grounded_rate >= 0.95
        and cross_tenant_leaks == 0
        and all(result.passed for result in results)
    )
    return EvaluationReport(
        tuple(results), retrieval_recall, grounded_rate, cross_tenant_leaks, p95_like_latency_ms, passed
    )


def build_demo_knowledge_base() -> TenantKnowledgeBase:
    knowledge_base = TenantKnowledgeBase(HashingEmbedder())
    knowledge_base.add(
        Document(
            "acme-deploy-v3",
            "acme",
            "Acme deployment policy requires staging evaluation before a canary rollout.",
            "3",
        )
    )
    knowledge_base.add(
        Document(
            "acme-tools-v1",
            "acme",
            "Acme tool policy requires least privilege and audit events for every execution.",
            "1",
        )
    )
    knowledge_base.add(
        Document(
            "globex-deploy-v7",
            "globex",
            "Globex deployment policy requires human approval before release.",
            "7",
        )
    )
    return knowledge_base


def demo() -> None:
    knowledge_base = build_demo_knowledge_base()
    responder = GroundedResponder(knowledge_base)

    answer = responder.answer("acme", "Which Acme deployment policy requires staging evaluation?")
    cross_tenant_query = knowledge_base.retrieve("acme", "Globex policy human approval", limit=2)
    cases = [
        EvaluationCase(
            "acme-deployment",
            "acme",
            "Which Acme deployment policy requires staging evaluation?",
            frozenset({"acme-deploy-v3"}),
        ),
        EvaluationCase(
            "acme-tools",
            "acme",
            "What does the Acme tool policy require?",
            frozenset({"acme-tools-v1"}),
        ),
    ]
    report = evaluate(responder, knowledge_base, cases)

    print("Grounded answer:", answer)
    print("Acme results for a Globex-looking query:", [item.document.document_id for item in cross_tenant_query])
    print("Evaluation report:", report)

    assert answer.citations == ("acme-deploy-v3",)
    assert all(item.document.tenant_id == "acme" for item in cross_tenant_query)
    assert report.passed
    print("PASS: RAG grounding, evaluation, and tenant-isolation checks completed.")


if __name__ == "__main__":
    demo()
