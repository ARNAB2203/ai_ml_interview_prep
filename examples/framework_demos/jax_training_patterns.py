"""JAX fundamentals: PRNG discipline, grad, jit, and vmap on CPU.

Run after installing `jax[cpu]`:
    python examples/framework_demos/jax_training_patterns.py
"""

# %% Imports
from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp


class Parameters(TypedDict):
    weights: jax.Array
    bias: jax.Array


def make_dataset(seed: int = 42) -> tuple[jax.Array, jax.Array]:
    """Split PRNG keys explicitly; unlike NumPy, JAX keys are single-use values."""
    key = jax.random.key(seed)
    feature_key, noise_key = jax.random.split(key)
    features = jax.random.normal(feature_key, (32, 2))
    targets = features @ jnp.array([[2.0], [-1.0]]) + 0.5 + 0.01 * jax.random.normal(noise_key, (32, 1))
    return features, targets


def predict(parameters: Parameters, features: jax.Array) -> jax.Array:
    return features @ parameters["weights"] + parameters["bias"]


def mean_squared_error(parameters: Parameters, features: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean((predict(parameters, features) - targets) ** 2)


per_example_squared_error = jax.vmap(
    lambda parameters, feature, target: jnp.squeeze((predict(parameters, feature[None, :]) - target[None, :]) ** 2),
    in_axes=(None, 0, 0),
)


@jax.jit
def update(parameters: Parameters, features: jax.Array, targets: jax.Array, learning_rate: float) -> Parameters:
    gradients = jax.grad(mean_squared_error)(parameters, features, targets)
    return {
        "weights": parameters["weights"] - learning_rate * gradients["weights"],
        "bias": parameters["bias"] - learning_rate * gradients["bias"],
    }


def train(steps: int = 30) -> dict[str, float]:
    features, targets = make_dataset()
    parameters: Parameters = {"weights": jnp.zeros((2, 1)), "bias": jnp.zeros((1,))}
    initial_loss = float(mean_squared_error(parameters, features, targets))
    for _ in range(steps):
        parameters = update(parameters, features, targets, 0.1)
    final_loss = float(mean_squared_error(parameters, features, targets))
    losses = per_example_squared_error(parameters, features, targets)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "max_per_example_loss": float(jnp.max(losses)),
    }


def demo() -> None:
    result = train()
    print("JAX result:", result)
    assert result["final_loss"] < result["initial_loss"]
    print("PASS: JAX PRNG, grad, jit, and vmap checks completed.")


if __name__ == "__main__":
    demo()
