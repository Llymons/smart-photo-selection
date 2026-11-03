"""Multi-cue scoring for candidate framings.

Four families of cues are combined:

* technical quality  – sharpness, exposure, contrast, colourfulness, noise;
* aesthetic quality  – a lightweight colour/tonal prior (plug-in interface
  for NIMA / MUSIQ / CLIP-aesthetic models);
* composition        – rule of thirds, saliency preservation, balance, border
  cuts and subject scale;
* semantic quality   – provided by an optional VLM judge (neutral by default).

All component scores are normalised to [0, 1] so that they can be fused and
compared across photos in the same series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from . import image_ops
from .types import Crop, ScoreBreakdown


@dataclass
class ScoringWeights:
    technical: float = 0.30
    aesthetic: float = 0.20
    composition: float = 0.30
    semantic: float = 0.20

    def normalised(self) -> "ScoringWeights":
        total = self.technical + self.aesthetic + self.composition + self.semantic
        if total <= 0:
            raise ValueError("Scoring weights must sum to a positive value")
        return ScoringWeights(
            technical=self.technical / total,
            aesthetic=self.aesthetic / total,
            composition=self.composition / total,
            semantic=self.semantic / total,
        )


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def technical_quality(image: np.ndarray) -> tuple[float, dict[str, float]]:
    """Rule-based technical quality in [0, 1]."""
    sharp_raw = image_ops.sharpness(image)
    sharp = _clip01((np.log10(sharp_raw + 1.0) - 1.0) / 2.0)

    mean_lum, clipped = image_ops.exposure_stats(image)
    exposure_mean = 1.0 - abs(mean_lum - 0.5) / 0.5
    exposure = 0.7 * _clip01(exposure_mean) + 0.3 * _clip01(1.0 - clipped / 0.08)

    contrast = _clip01(image_ops.contrast(image) / 0.30)
    colourful = _clip01(image_ops.colorfulness(image) / 60.0)
    noise_raw = image_ops.noise_level(image)
    noise = _clip01(1.0 - noise_raw / 15.0)

    score = (
        0.35 * sharp
        + 0.25 * exposure
        + 0.15 * contrast
        + 0.15 * colourful
        + 0.10 * noise
    )
    details = {
        "sharpness_raw": round(sharp_raw, 2),
        "exposure_mean": round(mean_lum, 4),
        "clipped_ratio": round(clipped, 4),
        "contrast_raw": round(image_ops.contrast(image), 4),
        "colorfulness_raw": round(image_ops.colorfulness(image), 2),
        "noise_raw": round(noise_raw, 2),
        "sharpness": round(sharp, 4),
        "exposure": round(exposure, 4),
    }
    return _clip01(score), details


def _thirds_alignment(cx: float, cy: float, w: float, h: float) -> float:
    rx, ry = cx / max(w, 1e-6), cy / max(h, 1e-6)
    dist = min(
        abs(rx - tx) + abs(ry - ty)
        for tx in (1 / 3, 2 / 3)
        for ty in (1 / 3, 2 / 3)
    )
    return float(np.exp(-(dist / 0.22) ** 2))


def composition_quality(
    saliency: np.ndarray,
    crop: Crop,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[float, dict[str, float]]:
    """Composition score for a crop, using the full-image saliency map."""
    h, w = saliency.shape[:2]
    x1, y1, x2, y2 = crop.box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    region = saliency[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0, {"saliency_mass": 0.0}

    total_mass = float(saliency.sum()) + 1e-8
    mass = float(region.sum()) / total_mass

    # Fraction of saliency close to the crop border -> risk of cutting subject.
    bw = max(2, int(round(crop.w * 0.06)))
    bh = max(2, int(round(crop.h * 0.06)))
    border = np.zeros_like(region, dtype=bool)
    border[:bh, :] = border[-bh:, :] = True
    border[:, :bw] = border[:, -bw:] = True
    border_mass = float(region[border].sum()) / (float(region.sum()) + 1e-8)
    border_score = 1.0 - _clip01(border_mass / 0.35)

    # Saliency centroid relative to the crop (rule of thirds).
    ys, xs = np.nonzero(saliency >= np.quantile(saliency, 0.8))
    if xs.size:
        cx = float(xs.mean()) - x1
        cy = float(ys.mean()) - y1
        thirds = _thirds_alignment(cx, cy, crop.w, crop.h)
        inside = 0.0 <= cx < crop.w and 0.0 <= cy < crop.h
        if not inside:
            thirds *= 0.25
    else:
        thirds = 0.5

    # Left/right and top/bottom balance of saliency.
    left = float(region[:, : region.shape[1] // 2].sum())
    right = float(region.sum()) - left
    balance_lr = 1.0 - abs(left - right) / (float(region.sum()) + 1e-8)
    top = float(region[: region.shape[0] // 2, :].sum())
    bottom = float(region.sum()) - top
    balance_tb = 1.0 - abs(top - bottom) / (float(region.sum()) + 1e-8)
    balance = 0.5 * balance_lr + 0.5 * balance_tb

    # Subject scale: avoid tiny subjects and extreme close-ups.
    if xs.size:
        sx1, sx2 = max(x1, int(xs.min())), min(x2, int(xs.max()) + 1)
        sy1, sy2 = max(y1, int(ys.min())), min(y2, int(ys.max()) + 1)
        subject_ratio = max(0, sx2 - sx1) * max(0, sy2 - sy1) / max(crop.area, 1)
        if subject_ratio < 0.05:
            subject = subject_ratio / 0.05
        elif subject_ratio > 0.75:
            subject = _clip01(1.0 - (subject_ratio - 0.75) / 0.25)
        else:
            subject = 1.0
    else:
        subject = 0.5

    # Faces near the crop border are usually a mistake.
    face_bonus = 1.0
    if face_boxes:
        for fx, fy, fw, fh in face_boxes:
            fcx, fcy = fx + fw / 2, fy + fh / 2
            if x1 <= fcx < x2 and y1 <= fcy < y2:
                margin = min(fcx - x1, x2 - fcx, fcy - y1, y2 - fcy)
                if margin < 0.04 * min(crop.w, crop.h):
                    face_bonus = min(face_bonus, 0.6)

    score = (
        0.30 * _clip01(mass / 0.75)
        + 0.25 * thirds
        + 0.15 * _clip01(balance)
        + 0.15 * subject
        + 0.15 * border_score
    ) * face_bonus
    details = {
        "saliency_mass": round(mass, 4),
        "thirds": round(thirds, 4),
        "balance": round(balance, 4),
        "subject_ratio": round(subject_ratio, 4) if xs.size else 0.0,
        "border_mass": round(border_mass, 4),
        "face_bonus": round(face_bonus, 4),
    }
    return _clip01(score), details


class AestheticScorer(Protocol):
    """Plug-in interface for learned image aesthetic assessment models."""

    name: str

    def score(self, image: np.ndarray) -> float:  # pragma: no cover - protocol
        ...


class HeuristicAestheticScorer:
    """Dependency-free aesthetic prior used until a learned model is wired in.

    It measures colour harmony and tonal appeal rather than semantics, so it is
    deliberately combined with composition and (optional) VLM judgements.
    """

    name = "heuristic-color-tonal"

    def score(self, image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = float(hsv[..., 1].mean() / 255.0)
        # Mid saturation is generally preferred to grey or oversaturated images.
        sat_score = float(np.exp(-((saturation - 0.45) / 0.30) ** 2))
        colourful = _clip01(image_ops.colorfulness(image) / 65.0)
        contrast = _clip01(image_ops.contrast(image) / 0.30)
        sharp = _clip01((np.log10(image_ops.sharpness(image) + 1.0) - 1.0) / 2.0)
        mean_lum, clipped = image_ops.exposure_stats(image)
        exposure = _clip01(1.0 - abs(mean_lum - 0.52) / 0.5) * _clip01(1.0 - clipped / 0.1)
        return _clip01(
            0.25 * sat_score
            + 0.25 * colourful
            + 0.20 * contrast
            + 0.15 * sharp
            + 0.15 * exposure
        )


def score_crop(
    image: np.ndarray,
    crop: Crop,
    saliency: np.ndarray,
    weights: ScoringWeights,
    aesthetic_scorer: AestheticScorer | None = None,
    semantic_score: float = 0.5,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> ScoreBreakdown:
    """Compute the full score breakdown for one candidate crop."""
    x1, y1, x2, y2 = crop.box
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return ScoreBreakdown(0.0, 0.0, 0.0, semantic_score, 0.0)

    # Scoring at a fixed maximum side keeps sharpness comparable across crops
    # of very different pixel sizes (e.g. a 16:9 crop vs. the full frame).
    patch = image_ops.resize_max(patch, 768)
    technical, tech_details = technical_quality(patch)
    scorer = aesthetic_scorer or HeuristicAestheticScorer()
    aesthetic = float(scorer.score(patch))
    composition, comp_details = composition_quality(saliency, crop, face_boxes)

    w = weights.normalised()
    total = (
        w.technical * technical
        + w.aesthetic * aesthetic
        + w.composition * composition
        + w.semantic * _clip01(semantic_score)
    )
    details = {"technical": tech_details, "composition": comp_details,
               "aesthetic_scorer": getattr(scorer, "name", "custom")}
    return ScoreBreakdown(
        technical=technical,
        aesthetic=aesthetic,
        composition=composition,
        semantic=_clip01(semantic_score),
        total=_clip01(total),
        details=details,
    )
