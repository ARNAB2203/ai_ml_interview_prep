"""A small tenant-aware Agent API implemented with FastAPI.

Run its in-process contract demonstration:
    python examples/agent_platform_api.py

Run as a local VS Code service after installing requirements:
    uvicorn agent_platform_api:app --app-dir examples --reload

Google Colab setup cell:
    %pip install -q -r examples/requirements.txt

The X-Tenant-ID header models a tenant claim already verified by an API gateway.
Never treat an arbitrary request header as authentication in a real deployment.
"""

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field


# %% Step 1 — versioned request and response contracts
class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_instruction: str = Field(min_length=10, max_length=4_000)
    allowed_tools: list[str] = Field(default_factory=list, max_length=10)
    retrieval_enabled: bool = True
    max_steps: int = Field(default=4, ge=1, le=10)


class AgentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=80, pattern=r"^[a-z][a-z0-9-]*$")
    config: AgentConfig


class AgentUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: AgentConfig


class AgentResponse(BaseModel):
    id: str
    name: str
    status: Literal["draft", "deployed"]
    version: int
    config: AgentConfig


class DeploymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: Literal["staging", "production"]
    evaluation_passed: bool


class DeploymentResponse(BaseModel):
    operation_id: str
    agent_id: str
    environment: str
    status: Literal["accepted"]


@dataclass
class AgentRecord:
    id: str
    tenant_id: str
    name: str
    config: AgentConfig
    status: Literal["draft", "deployed"] = "draft"
    version: int = 1

    def response(self) -> AgentResponse:
        return AgentResponse(
            id=self.id,
            name=self.name,
            status=self.status,
            version=self.version,
            config=self.config,
        )


# %% Step 2 — tenant scope is applied inside every storage operation
class AgentStore:
    """In-memory teaching store; production needs DB tenant controls and transactions."""

    def __init__(self) -> None:
        self._records: dict[str, AgentRecord] = {}

    def create(self, tenant_id: str, request: AgentCreateRequest) -> AgentRecord:
        record = AgentRecord(str(uuid4()), tenant_id, request.name, request.config)
        self._records[record.id] = record
        return record

    def list_for_tenant(self, tenant_id: str) -> list[AgentRecord]:
        return sorted(
            (record for record in self._records.values() if record.tenant_id == tenant_id),
            key=lambda record: record.name,
        )

    def get_for_tenant(self, tenant_id: str, agent_id: str) -> AgentRecord | None:
        record = self._records.get(agent_id)
        # Return None for another tenant to avoid resource enumeration.
        return record if record is not None and record.tenant_id == tenant_id else None


def tenant_context(x_tenant_id: str = Header(alias="X-Tenant-ID", min_length=3, max_length=64)) -> str:
    """Boundary placeholder for a gateway-verified tenant identity claim."""
    return x_tenant_id


def create_app(store: AgentStore | None = None) -> FastAPI:
    """Build an app with an injectable store so integration tests are isolated."""
    active_store = store or AgentStore()
    app = FastAPI(title="Agent Platform API", version="1.0.0")

    def require_agent(tenant_id: str, agent_id: str) -> AgentRecord:
        record = active_store.get_for_tenant(tenant_id, agent_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="agent not found")
        return record

    @app.post("/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
    def create_agent(request: AgentCreateRequest, tenant_id: str = Depends(tenant_context)) -> AgentResponse:
        return active_store.create(tenant_id, request).response()

    @app.get("/agents", response_model=list[AgentResponse])
    def list_agents(tenant_id: str = Depends(tenant_context)) -> list[AgentResponse]:
        return [record.response() for record in active_store.list_for_tenant(tenant_id)]

    @app.get("/agents/{agent_id}", response_model=AgentResponse)
    def get_agent(agent_id: str, tenant_id: str = Depends(tenant_context)) -> AgentResponse:
        return require_agent(tenant_id, agent_id).response()

    @app.put("/agents/{agent_id}", response_model=AgentResponse)
    def update_agent(
        agent_id: str,
        request: AgentUpdateRequest,
        tenant_id: str = Depends(tenant_context),
    ) -> AgentResponse:
        record = require_agent(tenant_id, agent_id)
        if record.status != "draft":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="deployed agents are immutable")
        record.config = request.config
        record.version += 1
        return record.response()

    @app.post("/agents/{agent_id}/deployments", response_model=DeploymentResponse, status_code=status.HTTP_202_ACCEPTED)
    def deploy_agent(
        agent_id: str,
        request: DeploymentRequest,
        tenant_id: str = Depends(tenant_context),
    ) -> DeploymentResponse:
        record = require_agent(tenant_id, agent_id)
        if request.environment == "production" and not request.evaluation_passed:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="production deployment requires evaluation")
        if record.status != "draft":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="agent is already deployed")
        record.status = "deployed"
        return DeploymentResponse(
            operation_id=str(uuid4()),
            agent_id=record.id,
            environment=request.environment,
            status="accepted",
        )

    return app


# App object is intentionally module-level for uvicorn.
app = create_app()


# %% Step 3 — in-process API demonstration and contract checks
def demo() -> None:
    client = TestClient(create_app())
    acme_headers = {"X-Tenant-ID": "acme"}
    agent_payload = {
        "name": "deployment-coach",
        "config": {
            "system_instruction": "Answer only from authorized deployment evidence.",
            "allowed_tools": ["search_knowledge"],
            "retrieval_enabled": True,
            "max_steps": 4,
        },
    }

    created = client.post("/agents", headers=acme_headers, json=agent_payload)
    agent_id = created.json()["id"]
    cross_tenant = client.get(f"/agents/{agent_id}", headers={"X-Tenant-ID": "globex"})
    blocked_deploy = client.post(
        f"/agents/{agent_id}/deployments",
        headers=acme_headers,
        json={"environment": "production", "evaluation_passed": False},
    )
    accepted_deploy = client.post(
        f"/agents/{agent_id}/deployments",
        headers=acme_headers,
        json={"environment": "production", "evaluation_passed": True},
    )

    print("Created agent:", created.json())
    print("Cross-tenant read status:", cross_tenant.status_code)
    print("Deployment without evaluation:", blocked_deploy.status_code, blocked_deploy.json())
    print("Accepted deployment:", accepted_deploy.json())

    assert created.status_code == status.HTTP_201_CREATED
    assert cross_tenant.status_code == status.HTTP_404_NOT_FOUND
    assert blocked_deploy.status_code == status.HTTP_409_CONFLICT
    assert accepted_deploy.status_code == status.HTTP_202_ACCEPTED
    print("PASS: Agent API contracts, tenant isolation, and deployment gate completed.")


if __name__ == "__main__":
    demo()
