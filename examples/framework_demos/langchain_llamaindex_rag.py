"""Credential-free LangChain and LlamaIndex RAG patterns.

Run after installing the framework CPU requirements:
    python examples/framework_demos/langchain_llamaindex_rag.py

This lab uses LangChain LCEL plus LlamaIndex's MockEmbedding. It proves the
framework wiring, metadata filter, citation contract, and tenant boundary
without contacting an LLM or embedding API.
"""

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KnowledgeDocument:
    source_id: str
    tenant_id: str
    text: str


DOCUMENTS = (
    KnowledgeDocument("acme-deploy-v3", "acme", "Acme requires staging evaluation before a canary rollout."),
    KnowledgeDocument("acme-tools-v1", "acme", "Acme requires least privilege and an audit event for every tool."),
    KnowledgeDocument("globex-deploy-v7", "globex", "Globex requires human approval before release."),
)


def local_tenant_retrieve(tenant_id: str, question: str) -> list[KnowledgeDocument]:
    """Tiny deterministic retriever used to make LCEL behavior inspectable."""
    keywords = set(question.lower().split())
    candidates = [document for document in DOCUMENTS if document.tenant_id == tenant_id]
    return sorted(
        candidates,
        key=lambda document: (-len(keywords.intersection(document.text.lower().split())), document.source_id),
    )


# %% Step 1 — LangChain LCEL composes deterministic ingress, retrieval, and response shaping
def run_langchain_chain(tenant_id: str, question: str) -> dict[str, Any]:
    from langchain_core.runnables import RunnableLambda

    def retrieve(payload: dict[str, str]) -> dict[str, Any]:
        documents = local_tenant_retrieve(payload["tenant_id"], payload["question"])
        return {**payload, "documents": documents}

    def ground(payload: dict[str, Any]) -> dict[str, Any]:
        documents: list[KnowledgeDocument] = payload["documents"]
        if not documents:
            return {"answer": "No authorized evidence found.", "citations": []}
        top = documents[0]
        return {"answer": top.text, "citations": [top.source_id]}

    chain = RunnableLambda(retrieve) | RunnableLambda(ground)
    return chain.invoke({"tenant_id": tenant_id, "question": question})


# %% Step 2 — LlamaIndex uses a metadata filter before retriever ranking
def run_llamaindex_retriever(tenant_id: str, question: str) -> list[dict[str, Any]]:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

    index = VectorStoreIndex.from_documents(
        [
            Document(
                text=document.text,
                metadata={"tenant_id": document.tenant_id, "source_id": document.source_id},
            )
            for document in DOCUMENTS
        ],
        embed_model=MockEmbedding(embed_dim=16),
    )
    filters = MetadataFilters(filters=[ExactMatchFilter(key="tenant_id", value=tenant_id)])
    retriever = index.as_retriever(similarity_top_k=2, filters=filters)
    return [
        {
            "source_id": item.node.metadata["source_id"],
            "tenant_id": item.node.metadata["tenant_id"],
            "score": float(item.score or 0),
        }
        for item in retriever.retrieve(question)
    ]


def demo() -> None:
    question = "What must Acme do before a canary rollout?"
    langchain_result = run_langchain_chain("acme", question)
    llamaindex_results = run_llamaindex_retriever("acme", question)
    print("LangChain LCEL result:", langchain_result)
    print("LlamaIndex metadata-filtered results:", llamaindex_results)
    assert langchain_result["citations"] == ["acme-deploy-v3"]
    assert llamaindex_results and all(item["tenant_id"] == "acme" for item in llamaindex_results)
    print("PASS: LangChain composition and LlamaIndex tenant filters completed.")


if __name__ == "__main__":
    demo()
