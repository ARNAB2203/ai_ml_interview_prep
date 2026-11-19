"""Provider-neutral model gateway with OpenAI and Anthropic adapter seams.

Default run is deterministic and makes no network request:
    python examples/framework_demos/provider_gateway.py

Install optional SDKs for live mode:
    python -m pip install openai anthropic

Set only one provider key securely, then opt in explicitly:
    OPENAI_API_KEY=... python examples/framework_demos/provider_gateway.py --live openai
    ANTHROPIC_API_KEY=... python examples/framework_demos/provider_gateway.py --live anthropic

The live path is intentionally never selected by default. It demonstrates an
adapter boundary; it does not print keys or place them in source control.
"""

# %% Imports
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from time import sleep
from typing import Any, Mapping, Protocol, Sequence


# %% Step 1 — a provider-independent contract
@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: tuple[Message, ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    max_tokens: int = 256


@dataclass(frozen=True)
class CompletionResponse:
    provider: str
    model: str
    text: str
    tool_calls: tuple[ToolCall, ...] = ()


class ModelClient(Protocol):
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Return normalized text and tool calls without leaking provider objects."""


class RetryableProviderError(RuntimeError):
    """Use for timeouts/rate limits that can be retried inside a bounded budget."""


def as_openai_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Translate the neutral tool contract to OpenAI's function-tool envelope."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


def as_anthropic_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Translate the neutral tool contract to Anthropic's tool-use envelope."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        }
        for tool in tools
    ]


def complete_with_retry(client: ModelClient, request: CompletionRequest, max_attempts: int = 2) -> CompletionResponse:
    """Retry only known-transient errors; policy errors must fail immediately."""
    for attempt in range(1, max_attempts + 1):
        try:
            return client.complete(request)
        except RetryableProviderError:
            if attempt == max_attempts:
                raise
            sleep(0.01 * attempt)  # Small deterministic teaching backoff; production adds jitter and deadlines.
    raise AssertionError("A provider call should return or raise.")


# %% Step 2 — fake provider enables tests without paid APIs or secrets
class FakeModelClient:
    """Deterministic test double that emits a structured, policy-safe response."""

    def __init__(self, fail_once: bool = False) -> None:
        self._fail_once = fail_once
        self.calls = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if self._fail_once and self.calls == 1:
            raise RetryableProviderError("simulated rate limit")
        user_text = next((message.content for message in reversed(request.messages) if message.role == "user"), "")
        if "capacity" in user_text.lower():
            return CompletionResponse(
                provider="fake",
                model=request.model,
                text="I will use the approved capacity calculator.",
                tool_calls=(ToolCall("calculate_capacity", {"requests_per_second": 12, "p95_seconds": 3}),),
            )
        return CompletionResponse("fake", request.model, "I need authorized evidence before answering.")


# %% Step 3 — live adapters normalize SDK-specific message/tool formats
class OpenAIChatClient:
    """Thin adapter for the OpenAI Chat Completions surface.

    Keep this adapter small so a provider migration changes one boundary instead
    of every FastAPI route, graph node, or evaluation test.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_environment(cls) -> "OpenAIChatClient":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required only for --live openai mode.")
        from openai import OpenAI  # Lazy import keeps fake mode dependency-light.

        return cls(OpenAI())

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            response = self._client.chat.completions.create(
                model=request.model,
                messages=[{"role": message.role, "content": message.content} for message in request.messages],
                tools=as_openai_tools(request.tools) or None,
                max_tokens=request.max_tokens,
                temperature=0,
            )
        except Exception as exc:  # SDK-specific exceptions should be classified more precisely in production.
            raise RetryableProviderError("OpenAI request failed; inspect provider error classification.") from exc

        message = response.choices[0].message
        tool_calls = tuple(
            ToolCall(call.function.name, json.loads(call.function.arguments))
            for call in (message.tool_calls or [])
        )
        return CompletionResponse("openai", request.model, message.content or "", tool_calls)


class AnthropicMessagesClient:
    """Thin adapter for Anthropic Messages/tool-use responses."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_environment(cls) -> "AnthropicMessagesClient":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required only for --live anthropic mode.")
        from anthropic import Anthropic  # Lazy import keeps fake mode dependency-light.

        return cls(Anthropic())

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        system = "\n".join(message.content for message in request.messages if message.role == "system")
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role != "system"
        ]
        try:
            response = self._client.messages.create(
                model=request.model,
                system=system or None,
                messages=messages,
                tools=as_anthropic_tools(request.tools) or None,
                max_tokens=request.max_tokens,
            )
        except Exception as exc:
            raise RetryableProviderError("Anthropic request failed; inspect provider error classification.") from exc

        text_blocks = [block.text for block in response.content if block.type == "text"]
        tool_calls = tuple(
            ToolCall(block.name, dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        )
        return CompletionResponse("anthropic", request.model, "\n".join(text_blocks), tool_calls)


def demo_request(model: str = "demo-model") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=(
            Message("system", "Use only approved tools and return concise plans."),
            Message("user", "Estimate capacity for this agent API."),
        ),
        tools=(
            {
                "name": "calculate_capacity",
                "description": "Compute concurrent requests from safe numeric inputs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "requests_per_second": {"type": "number", "minimum": 0},
                        "p95_seconds": {"type": "number", "minimum": 0},
                    },
                    "required": ["requests_per_second", "p95_seconds"],
                    "additionalProperties": False,
                },
            },
        ),
    )


def demo() -> None:
    client = FakeModelClient(fail_once=True)
    response = complete_with_retry(client, demo_request())
    print("Provider:", response.provider)
    print("Text:", response.text)
    print("Structured tool call:", response.tool_calls)
    assert client.calls == 2
    assert response.tool_calls[0].name == "calculate_capacity"
    print("PASS: provider-neutral retry and structured-tool contracts completed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", choices=("openai", "anthropic"), help="Opt in to a credentialed provider request.")
    arguments = parser.parse_args()
    if arguments.live == "openai":
        print(OpenAIChatClient.from_environment().complete(demo_request("gpt-4.1-mini")))
    elif arguments.live == "anthropic":
        print(AnthropicMessagesClient.from_environment().complete(demo_request("claude-sonnet-4-20250514")))
    else:
        demo()


if __name__ == "__main__":
    main()
