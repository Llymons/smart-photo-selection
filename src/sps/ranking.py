"""Within-series ranking and a small learning-to-rank module.

The final decision is a *ranking* problem: given the best framing of each photo
in a burst, which photo would a person keep?  We provide:

* percentile normalisation of each cue inside the series (removes global bias
  such as a whole burst being slightly dark);
* a Bradley-Terry / RankNet-style pairwise logistic ranker trained on human
  pairwise preferences, implemented in NumPy (no scikit-learn dependency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import PhotoResult

FEATURE_NAMES = ("technical", "aesthetic", "composition", "semantic")


def feature_vector(photo: PhotoResult) -> np.ndarray:
    s = photo.best.score
    return np.array([s.technical, s.aesthetic, s.composition, s.semantic],
                    dtype=np.float64)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-rank percentile in [0, 1] (ties share the same rank)."""
    n = len(values)
    if n <= 1:
        return np.full(n, 0.5)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Average ties.
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks / (n - 1)


def series_scores(
    photos: list[PhotoResult],
    weights,
    ranker: "PairwiseRanker | None" = None,
) -> np.ndarray:
    """Score all photos in a series on a common scale."""
    if not photos:
        return np.array([])
    features = np.stack([feature_vector(p) for p in photos])
    if ranker is not None:
        return ranker.predict(features)

    # Relative cues: give each photo its percentile within the burst.
    normalised = np.stack(
        [_percentile_rank(features[:, i]) for i in range(features.shape[1])],
        axis=1,
    )
    w = weights.normalised()
    w_vec = np.array([w.technical, w.aesthetic, w.composition, w.semantic])
    return normalised @ w_vec


@dataclass
class PairwiseRanker:
    """Pairwise logistic regression (Bradley-Terry) over cue features."""

    learning_rate: float = 0.25
    epochs: int = 800
    l2: float = 1e-3
    weights: np.ndarray | None = None

    def fit(self, preferred: np.ndarray, other: np.ndarray) -> "PairwiseRanker":
        preferred = np.asarray(preferred, dtype=np.float64)
        other = np.asarray(other, dtype=np.float64)
        if preferred.shape != other.shape:
            raise ValueError("Pairwise feature matrices must have the same shape")
        n_features = preferred.shape[1]
        w = np.zeros(n_features, dtype=np.float64)
        diff = preferred - other

        for _ in range(self.epochs):
            logits = diff @ w
            # Stable sigmoid.
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            grad = -(diff * (1.0 - prob)[:, None]).mean(axis=0) + self.l2 * w
            w -= self.learning_rate * grad
        self.weights = w
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("PairwiseRanker is not fitted")
        return np.asarray(features, dtype=np.float64) @ self.weights

    def save(self, path: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("PairwiseRanker is not fitted")
        Path(path).write_text(
            json.dumps({"features": list(FEATURE_NAMES),
                        "weights": self.weights.tolist()}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseRanker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ranker = cls()
        ranker.weights = np.asarray(data["weights"], dtype=np.float64)
        return ranker


def rank_photos(
    photos: list[PhotoResult],
    weights,
    ranker: PairwiseRanker | None = None,
) -> list[PhotoResult]:
    """Assign series scores and return photos sorted best-first."""
    scores = series_scores(photos, weights, ranker)
    for photo, score in zip(photos, scores):
        photo.series_score = float(score)
    ranked = sorted(photos, key=lambda p: p.series_score, reverse=True)
    for i, photo in enumerate(ranked, start=1):
        photo.rank = i
    return ranked
