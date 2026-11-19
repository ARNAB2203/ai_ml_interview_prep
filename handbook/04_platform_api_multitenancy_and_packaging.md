# Module 04 — Agent Platform API, Multi-tenancy, and Packaging

**Level:** Advanced  
**Interview goal:** Design the control plane that safely manages many agents,
not just one successful prompt.

## 1. Resource model and lifecycle

An Agent API should distinguish configuration, immutable build artifact,
deployment, run, and evaluation result. Collapsing them into one mutable “agent”
object makes reproducibility and rollback difficult.

```text
Agent (logical identity)
  ├── AgentVersion (immutable configuration + prompt + schema)
  ├── EvaluationRun (dataset + scores + trace links)
  ├── PackageArtifact (digest + manifest + provenance)
  └── Deployment (environment + desired/actual state + rollout)
```

Useful endpoints include:

| Endpoint | Intent | Important contract |
| --- | --- | --- |
| `POST /agents` | Create a draft | Idempotency key; schema validation; tenant scope |
| `GET /agents/{id}` | Read a resource | Tenant authorization; no cross-tenant enumeration |
| `PUT /agents/{id}` | Update draft | Optimistic version / ETag; audit change set |
| `POST /agents/{id}/evaluations` | Start an evaluation | Dataset/version and runtime are captured |
| `POST /agents/{id}/deployments` | Request deployment | Gate/approval, immutable version, async operation |
| `GET /operations/{id}` | Poll an async change | Correlation ID and clear terminal states |

## 2. Tenant isolation is a system property

Do not trust a `tenant_id` supplied in a JSON body. Derive it from verified
identity or a gateway claim, propagate it through every query/job/cache key, and
enforce it in the data layer.

| Layer | Isolation control |
| --- | --- |
| API gateway | Verify token issuer/audience; map subject to tenant and scopes |
| Application | Inject tenant context; authorize resource action |
| Database | Tenant predicates plus row-level security or separate schema/database as appropriate |
| Cache / queue | Prefix keys, include tenant in messages, authenticate consumers |
| Retrieval | Pre-filter by tenant and document ACL |
| Telemetry | Tenant-aware access control; redact payloads; retention policy |

> **Warning:** An application-level `WHERE tenant_id = ?` alone is vulnerable
> to a missed query. Defense in depth is essential for a platform control plane.

## 3. Concept → code: typed, tenant-aware Agent API

```bash
python examples/agent_platform_api.py
```

The FastAPI example runs in-process through `TestClient`; it neither opens a
port nor needs a database. It demonstrates typed request models, a tenant
header dependency, tenant-scoped list/get/update behavior, a configuration
schema, and a guarded `draft → deployed` transition.

To run it as a local development service after installation:

```bash
uvicorn agent_platform_api:app --app-dir examples --reload
```

Then add a real persistence layer with database-level tenant controls,
transactions/outbox, rate limits, authorization claims, and asynchronous
deployment workers. Do not use the in-memory store in production.

## 4. Package and manifest validation

Custom/code-based agents need a package contract before execution. A validator
should inspect an archive without executing it.

| Validation | Why it matters |
| --- | --- |
| Safe archive paths | Blocks `../` traversal and absolute-path writes |
| Entry count / unpacked-size limits | Limits decompression bombs and resource abuse |
| Required manifest fields | Identifies entry point, protocol, runtime, and version |
| Schema and compatibility checks | Prevents unsupported contracts at runtime |
| Digest/provenance/signature | Connects deployment to an immutable reviewed artifact |
| Dependency policy / SBOM scan | Governs vulnerable or prohibited dependencies |

```bash
python examples/manifest_validator.py
```

The example creates a safe demo package and rejects a traversal path. In a real
platform, upload to quarantine storage, scan and validate asynchronously,
produce an immutable artifact digest, and execute only within a restrictive
sandbox.

## 5. A2A/MCP-style interoperability discussion

Protocol names are less important than stable contracts. Ask:

- How is a capability discovered and versioned?
- What input/output schema and error taxonomy are guaranteed?
- How are caller identity, delegated authority, and tenant context propagated?
- What are timeout, cancellation, streaming, and retry semantics?
- How are tools constrained so an integration server cannot become an
  unrestricted exfiltration or execution channel?

For an A2A-style message, include message ID, correlation ID, protocol version,
sender/recipient identity, tenant context, typed payload, and terminal outcome.
For MCP-style tools, expose narrow typed operations and independently enforce
authorization on the server.

## 6. Interview drill: design a custom-agent upload flow

```text
browser/API → authenticated upload URL → quarantine object storage
      → async validator / malware + manifest + policy checks
      → immutable artifact + digest + evaluation request
      → approval / rollout controller → isolated runtime
```

Mention retry-safe operation IDs, status visibility, content-size limits,
signed upload URLs, event/audit records, and revocation/rollback. A senior
answer explicitly says what never executes during validation.

Proceed to [Module 05: Reliability, Security, and Deployment](05_production_reliability_security_and_deployment.md).
