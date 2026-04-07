"""Shared text helpers for lightweight semantic scoring."""

from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a scalar value into a closed interval."""
    return max(low, min(high, value))


def normalize_text(text: str) -> str:
    """Lower-case and collapse whitespace for stable comparisons."""
    return " ".join(text.lower().split())


def tokenize(text: str) -> list[str]:
    """Tokenize text into a lightweight alphanumeric bag."""
    return TOKEN_RE.findall(normalize_text(text))


def hashed_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Create a deterministic hashed bag-of-words embedding."""
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=2).hexdigest()
        index = int(digest, 16) % dim
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity with safe zero-vector handling."""
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def text_similarity(left: str, right: str) -> float:
    """Compute deterministic cosine similarity between two texts."""
    return cosine_similarity(hashed_embedding(left), hashed_embedding(right))


def stable_noise(text: str, low: float = -0.05, high: float = 0.05) -> float:
    """Map a text fingerprint to a stable uniform noise value."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    value = int(digest, 16) / float(16**16 - 1)
    return low + (high - low) * value

