"""Tokenizer and byte-pair encoding (BPE) lesson.

Run locally:
    python examples/tokenizer_bpe.py

Google Colab setup cell:
    # No third-party package is required for this lesson.

The example mirrors the supplied notebook progression: tokenize a small string,
map tokens to IDs, make unknown-token behavior explicit, then learn simple BPE
merges. It is educational code, not a replacement for a model's official
tokenizer.
"""

# %% Imports
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re


# %% Step 1 — split text into inspectable tokens
TOKEN_PATTERN = re.compile(r"<\|[^|]+\|>|--|[A-Za-z0-9_]+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Return words, special tokens, and punctuation while dropping whitespace."""
    return TOKEN_PATTERN.findall(text)


# %% Step 2 — turn tokens into integer IDs
@dataclass
class SimpleTokenizerV1:
    """A fixed vocabulary tokenizer that raises for an unknown token."""

    str_to_int: dict[str, int]
    int_to_str: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.int_to_str = {token_id: token for token, token_id in self.str_to_int.items()}

    @classmethod
    def from_corpus(cls, corpus: str) -> "SimpleTokenizerV1":
        vocabulary = {token: index for index, token in enumerate(sorted(set(tokenize(corpus))))}
        return cls(vocabulary)

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int[token] for token in tokenize(text)]

    def decode(self, token_ids: list[int]) -> str:
        text = " ".join(self.int_to_str[token_id] for token_id in token_ids)
        return re.sub(r"\s+([,.:;?!'\")])", r"\1", text)


@dataclass
class SimpleTokenizerV2(SimpleTokenizerV1):
    """Version 2 adds an explicit unknown-token policy."""

    unknown_token: str = "<unk>"

    def encode(self, text: str) -> list[int]:
        unknown_id = self.str_to_int[self.unknown_token]
        return [self.str_to_int.get(token, unknown_id) for token in tokenize(text)]


# %% Step 3 — learn BPE merges from a tiny corpus
Symbol = tuple[str, ...]
Pair = tuple[str, str]


@dataclass
class BPETrainer:
    """Small, deterministic character-level BPE trainer for interview practice.

    Production tokenizers include byte handling, normalization, special-token
    policy, and highly optimized implementations. Keeping this version small
    makes the merge rule visible and testable.
    """

    merges: list[Pair] = field(default_factory=list)

    @staticmethod
    def _symbols(word: str) -> Symbol:
        return tuple(word) + ("</w>",)

    @staticmethod
    def _pair_counts(vocabulary: Counter[Symbol]) -> Counter[Pair]:
        counts: Counter[Pair] = Counter()
        for symbols, frequency in vocabulary.items():
            counts.update({(symbols[index], symbols[index + 1]): frequency for index in range(len(symbols) - 1)})
        return counts

    @staticmethod
    def _merge_pair(symbols: Symbol, pair: Pair) -> Symbol:
        merged: list[str] = []
        index = 0
        while index < len(symbols):
            if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                merged.append("".join(pair))
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        return tuple(merged)

    def train(self, corpus: str, num_merges: int = 12) -> list[Pair]:
        """Learn the most frequent pair at each step, with deterministic ties."""
        words = [token.lower() for token in tokenize(corpus) if token.isalnum()]
        vocabulary: Counter[Symbol] = Counter(self._symbols(word) for word in words)
        self.merges = []

        for _ in range(num_merges):
            pair_counts = self._pair_counts(vocabulary)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            self.merges.append(best_pair)
            vocabulary = Counter(
                {self._merge_pair(symbols, best_pair): frequency for symbols, frequency in vocabulary.items()}
            )
        return self.merges

    def encode_word(self, word: str) -> list[str]:
        symbols = self._symbols(word.lower())
        for pair in self.merges:
            symbols = self._merge_pair(symbols, pair)
        return list(symbols)

    def encode_text(self, text: str) -> list[str]:
        pieces: list[str] = []
        for token in tokenize(text):
            pieces.extend(self.encode_word(token) if token.isalnum() else [token])
        return pieces


def demo() -> None:
    """Run the lesson with explicit inputs and observable output."""
    sample = "Hello, agent platform!"
    print("Step 1 — tokens:", tokenize(sample))

    corpus = "Agents plan carefully. An agent uses tools safely."
    v1 = SimpleTokenizerV1.from_corpus(corpus)
    known_ids = v1.encode("Agents plan carefully.")
    print("\nStep 2 — vocabulary size:", len(v1.str_to_int))
    print("Known token IDs:", known_ids)
    print("Decoded known text:", v1.decode(known_ids))

    v2_vocab = dict(v1.str_to_int)
    v2_vocab["<unk>"] = len(v2_vocab)
    v2 = SimpleTokenizerV2(v2_vocab)
    unknown_ids = v2.encode("Agents orchestrate safely.")
    print("Unknown-safe IDs:", unknown_ids)
    print("Unknown-safe decode:", v2.decode(unknown_ids))

    bpe = BPETrainer()
    merges = bpe.train("agent agents agentic platform platform tenant", num_merges=10)
    print("\nStep 3 — learned BPE merges:", merges)
    print("BPE pieces for 'agentic':", bpe.encode_word("agentic"))
    print("BPE pieces for 'agent-platform':", bpe.encode_text("agent-platform"))

    # Expected behavior: known text round-trips, and unseen words do not raise.
    assert v1.decode(known_ids) == "Agents plan carefully."
    assert v2.str_to_int["<unk>"] in unknown_ids
    assert bpe.encode_word("agentic")
    print("\nPASS: tokenizer and BPE checks completed.")


if __name__ == "__main__":
    demo()
