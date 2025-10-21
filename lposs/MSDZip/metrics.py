"""Utilities for recording per-symbol compression metrics."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


class CompressionMetricsRecorder:
    """Collects bitrate metrics during compression.

    The recorder stores the number of bits spent on each symbol in the
    original sequence. Positions that were not observed remain ``NaN`` to
    make it easy to mask them out when performing downstream analysis.
    """

    def __init__(self, length: int, vocab_size: int) -> None:
        self.length = int(length)
        self.vocab_size = int(vocab_size)
        self.bits = np.full(self.length, np.nan, dtype=np.float32)
        self._uniform_bits = math.log2(self.vocab_size) if self.vocab_size > 0 else 0.0

    def _sanitize_positions(self, positions: Sequence[int]) -> np.ndarray:
        arr = np.asarray(positions, dtype=np.int64)
        if arr.size == 0:
            return arr
        mask = (arr >= 0) & (arr < self.length)
        return arr[mask]

    def record_uniform(self, positions: Iterable[int], overwrite: bool = False) -> None:
        """Record symbols encoded with an equiprobable model.

        Args:
            positions: Iterable of indices in the original sequence.
            overwrite: Whether to overwrite existing measurements. By default
                only previously-unseen positions are filled.
        """

        arr = self._sanitize_positions(list(positions))
        if arr.size == 0:
            return
        if overwrite:
            self.bits[arr] = self._uniform_bits
        else:
            mask = np.isnan(self.bits[arr])
            if mask.any():
                self.bits[arr[mask]] = self._uniform_bits

    def record_probabilities(self, positions: Sequence[int], probabilities: Sequence[float]) -> None:
        """Record bitrate for symbols predicted by the neural model."""

        arr = self._sanitize_positions(list(positions))
        if arr.size == 0:
            return
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape[0] != arr.shape[0]:
            raise ValueError("positions and probabilities must have the same length")
        safe_probs = np.clip(probs, 1e-12, 1.0)
        bits = (-np.log2(safe_probs)).astype(np.float32)
        mask = np.isnan(self.bits[arr])
        if mask.any():
            self.bits[arr[mask]] = bits[mask]

    def topk(self, k: int) -> List[Tuple[int, float]]:
        """Return the hardest ``k`` positions sorted by bitrate."""

        if k <= 0:
            return []
        valid_mask = ~np.isnan(self.bits)
        indices = np.where(valid_mask)[0]
        if indices.size == 0:
            return []
        values = self.bits[indices]
        order = np.argsort(values)[::-1]
        order = order[:k]
        return [(int(indices[i]), float(values[i])) for i in order]

    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        """Persist the recorded metrics to disk."""

        np.savez(path, bits=self.bits, metadata=metadata or {})
