"""Production-shaped FastAPI model-serving boundary with health and metrics.

Run the in-process smoke test:
    python examples/ml_ds_demos/model_serving_api.py

Run a local service:
    uvicorn model_serving_api:app --app-dir examples/ml_ds_demos --host 0.0.0.0 --port 8010

The score is deterministic teaching logic. The API concerns—startup, readiness,
validation, request IDs, metrics, bounded response schema, and testability—are
the same concerns for a loaded model artifact.
"""

# %% Imports
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_ms: float = Field(ge=0, le=60_000)
    error_count: int = Field(ge=0, le=1_000)
    region: str = Field(pattern=r"^(us|eu|apac)$")


class PredictionResponse(BaseModel):
    model_version: str
    risk_probability: float
    predicted_incident: bool
    request_id: str


@dataclass
class Metrics:
    total_requests: int = 0
    total_predict_requests: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def record(self, path: str, duration_ms: float) -> None:
        self.total_requests += 1
        self.latency_ms.append(duration_ms)
        if path == "/predict":
            self.total_predict_requests += 1

    def prometheus(self) -> str:
        p95_index = max(0, int(0.95 * len(self.latency_ms)) - 1)
        p95 = sorted(self.latency_ms)[p95_index] if self.latency_ms else 0.0
        return "\n".join(
            [
                "# TYPE model_requests_total counter",
                f"model_requests_total {self.total_requests}",
                "# TYPE model_predict_requests_total counter",
                f"model_predict_requests_total {self.total_predict_requests}",
                "# TYPE model_request_latency_p95_ms gauge",
                f"model_request_latency_p95_ms {p95:.3f}",
                "",
            ]
        )


@dataclass
class ModelRuntime:
    model_version: str = "risk-model:demo-1.0.0"
    ready: bool = False
    metrics: Metrics = field(default_factory=Metrics)

    def load(self) -> None:
        # Replace with verified artifact load + schema/tokenizer compatibility checks.
        self.ready = True

    def predict(self, payload: PredictionRequest) -> float:
        regional_adjustment = {"us": 0.00, "eu": 0.04, "apac": 0.08}[payload.region]
        probability = 0.03 + min(payload.duration_ms / 20_000, 0.45) + min(payload.error_count * 0.09, 0.40) + regional_adjustment
        return round(min(probability, 0.99), 4)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = ModelRuntime()
        runtime.load()
        app.state.runtime = runtime
        yield
        runtime.ready = False

    app = FastAPI(title="Model Serving API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_metrics(request: Request, call_next):
        started = perf_counter()
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        runtime: ModelRuntime = request.app.state.runtime
        runtime.metrics.record(request.url.path, (perf_counter() - started) * 1_000)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        runtime: ModelRuntime = request.app.state.runtime
        if not runtime.ready:
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return {"status": "ready", "model_version": runtime.model_version}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        runtime: ModelRuntime = request.app.state.runtime
        if not runtime.ready:
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        probability = runtime.predict(payload)
        return PredictionResponse(
            model_version=runtime.model_version,
            risk_probability=probability,
            predicted_incident=probability >= 0.50,
            request_id=request.state.request_id,
        )

    @app.get("/metrics", response_class=Response)
    async def metrics(request: Request) -> Response:
        runtime: ModelRuntime = request.app.state.runtime
        return Response(content=runtime.metrics.prometheus(), media_type="text/plain; version=0.0.4")

    return app


app = create_app()


def demo() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")
        prediction = client.post(
            "/predict",
            headers={"X-Request-ID": "demo-request-42"},
            json={"duration_ms": 8_000, "error_count": 3, "region": "apac"},
        )
        invalid = client.post("/predict", json={"duration_ms": -1, "error_count": 0, "region": "us"})
        metrics = client.get("/metrics")

    print("Health:", health.json())
    print("Ready:", ready.json())
    print("Prediction:", prediction.json())
    print("Invalid request status:", invalid.status_code)
    print("Metrics:\n", metrics.text)
    assert health.status_code == 200 and ready.status_code == 200
    assert prediction.status_code == 200 and prediction.json()["request_id"] == "demo-request-42"
    assert invalid.status_code == 422
    assert "model_predict_requests_total 2" in metrics.text
    print("PASS: serving API health, validation, prediction, request-ID, and metrics checks completed.")


if __name__ == "__main__":
    demo()
