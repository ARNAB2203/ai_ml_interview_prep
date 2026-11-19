# AI/ML Interview Preparation Handbook — Agentic Platform Engineering

This repository contains a PDF-ready, senior-level interview preparation handbook
tailored to the supplied **Senior Agentic Platform Engineer (Python/ML)** job
description. It is aimed at a candidate with up to 12 years of software and
architecture experience and 3+ years of AI/ML or GenAI engineering experience.

The material deliberately connects concepts to working code. The goal is not
just to recall terminology; it is to explain and implement the trade-offs that
matter in a production, multi-tenant agent platform.

## Start here

1. Read [the overview and study plan](handbook/00_overview_and_study_plan.md).
2. Work through the handbook in numerical order.
3. Run the matching example after each topic. By default, every example uses
   deterministic sample data, makes no external model call, and is safe to run
   without API keys; clearly labelled live-provider paths are opt-in.
4. Use the interview drills to practice an answer aloud and then implement a
   small vertical slice under a time limit.

### Local VS Code setup

```bash
cd ai_ml_interview_prep
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r examples/requirements.txt
python examples/run_all.py
python -m unittest discover -s examples/tests -v
```

Open any `examples/*.py` file in VS Code and run its `# %%` cells, or run the
file as a normal Python script. The examples use the same small-step,
explanation-first progression as the supplied tokenizer and BPE notebooks.

### Optional framework and ML stack (CPU-safe)

The framework track adds real, local LangChain, LlamaIndex, LangGraph, MCP,
PyTorch, JAX, Hugging Face, and LoRA examples. It is deliberately separate from
the lightweight baseline so a basic revision session does not install model
frameworks or accelerator packages unnecessarily.

```bash
python -m venv .framework-venv
source .framework-venv/bin/activate     # Windows: .framework-venv\\Scripts\\activate
python -m pip install -r examples/requirements-frameworks-cpu.txt
python examples/framework_demos/run_cpu_demos.py
python -m unittest discover -s examples/framework_demos/tests -v
```

OpenAI, Anthropic, AutoGen, CrewAI, Google ADK, QLoRA, and DeepSpeed live/GPU
paths are opt-in. See Modules 08–11 before installing the corresponding
optional dependency: the code defaults to local test doubles, an offline tiny
model, or configuration validation and never requires a key or GPU by default.

### Google Colab setup

Create a new Colab notebook and run these cells in order. The repository URL
and branch are intentionally explicit so the notebook is reproducible before
the PR is merged.

```python
# Cell 1 — obtain the handbook
!git clone https://github.com/ARNAB2203/ai_ml_interview_prep.git
%cd ai_ml_interview_prep
!git checkout feature/ai-ml-interview-prep-handbook
```

```python
# Cell 2 — install the only third-party packages used by the API example
%pip install -q -r examples/requirements.txt
```

```python
# Cell 3 — run one self-contained lesson (change the filename for another)
!python examples/tokenizer_bpe.py
```

For a notebook-form runner with installation, imports, and expected output,
open [notebooks/00_colab_runner.ipynb](notebooks/00_colab_runner.ipynb) in
Colab after cloning the repository. The stack-extension notebooks are
[01_frameworks_colab.ipynb](notebooks/01_frameworks_colab.ipynb) and
[02_model_engineering_colab.ipynb](notebooks/02_model_engineering_colab.ipynb).

## Handbook map

| Level | Topic | Primary runnable artifact |
| --- | --- | --- |
| Beginner | Production Python, ML foundations, tokens and BPE | `examples/tokenizer_bpe.py` |
| Intermediate | Prompts, tools, state, and agent orchestration | `examples/tool_using_agent.py` |
| Intermediate | Retrieval-augmented generation and measurable quality | `examples/rag_evaluation.py` |
| Advanced | Agent lifecycle APIs, schemas, and tenant isolation | `examples/agent_platform_api.py` |
| Advanced | Package manifests, upload safety, and interoperability | `examples/manifest_validator.py` |
| Master | Reliability, security, deployment, evaluation gates, and system design | Handbook modules 05–07 |
| Advanced | LangChain, LlamaIndex, OpenAI/Anthropic adapter boundaries, and FastAPI integration | `examples/framework_demos/provider_gateway.py` |
| Advanced | LangGraph, MCP, AutoGen, CrewAI, and Google ADK decision patterns | `examples/framework_demos/langgraph_approval_graph.py` |
| Advanced | PyTorch, JAX, Hugging Face, PEFT/LoRA/QLoRA, and DeepSpeed | `examples/framework_demos/pytorch_hf_peft.py` |

## Repository layout

```text
handbook/                 # PDF-ready Markdown modules
examples/                 # Self-contained Python implementations and tests
notebooks/                # Colab/VS Code Jupyter runner
configs/                  # Production-oriented configuration examples
```

The handbook does not claim that its deterministic examples are a substitute
for a production LLM provider, vector database, identity provider, or cloud
control plane. Each module calls out the boundary and the controls required
when making that substitution in a real service.
