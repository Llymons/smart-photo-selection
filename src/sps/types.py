"""Core data structures used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Crop:
    """An axis-aligned crop window in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2)"""
        return self.x, self.y, self.x + self.w, self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h

    def iou(self, other: "Crop") -> float:
        ax1, ay1, ax2, ay2 = self.box
        bx1, by1, bx2, by2 = other.box
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class ScoreBreakdown:
    """Scores are normalised to [0, 1] unless stated otherwise."""

    technical: float
    aesthetic: float
    composition: float
    semantic: float
    total: float
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "technical": round(self.technical, 4),
            "aesthetic": round(self.aesthetic, 4),
            "composition": round(self.composition, 4),
            "semantic": round(self.semantic, 4),
            "total": round(self.total, 4),
            "details": self.details,
        }


@dataclass
class CandidateResult:
    crop: Crop
    score: ScoreBreakdown


@dataclass
class PhotoResult:
    path: Path
    image_size: tuple[int, int]  # (width, height)
    best: CandidateResult
    candidates: list[CandidateResult]
    series_score: float = 0.0
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "image_size": list(self.image_size),
            "best_crop": self.best.crop.as_dict(),
            "best_score": self.best.score.as_dict(),
            "series_score": round(self.series_score, 4),
            "rank": self.rank,
            "num_candidates": len(self.candidates),
        }


@dataclass
class SeriesResult:
    photos: list[PhotoResult]
    selected: PhotoResult
    ranking: list[PhotoResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": str(self.selected.path),
            "ranking": [p.as_dict() for p in self.ranking],
        }
