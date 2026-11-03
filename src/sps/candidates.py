"""Candidate framing (crop) generation.

The goal is *not* to enumerate every possible crop, but to produce a compact,
diverse set of plausible framings that covers:

1. common photographic aspect ratios;
2. multiple zoom levels (scale of the crop relative to the image);
3. rule-of-thirds placements of the salient subject;
4. grid positions that keep context around the subject.

Candidates are de-duplicated greedily by IoU and pre-ranked by a cheap
composition prior so that the expensive scorers only see useful crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from .types import Crop


@dataclass
class CandidateConfig:
    aspect_ratios: tuple[float, ...] = (1.0, 4 / 3, 3 / 2, 16 / 9)
    scales: tuple[float, ...] = (0.55, 0.72, 0.88, 1.0)
    grid_steps: int = 3
    max_candidates: int = 80
    iou_threshold: float = 0.92
    min_side: int = 96


def saliency_centroid(saliency: np.ndarray, top_ratio: float = 0.2) -> tuple[float, float] | None:
    """Centroid of the strongest saliency region, in pixel coordinates."""
    flat = saliency.reshape(-1)
    if flat.size == 0:
        return None
    threshold = np.quantile(flat, 1.0 - top_ratio)
    ys, xs = np.nonzero(saliency >= threshold)
    if xs.size == 0:
        return None
    weights = saliency[ys, xs]
    total = float(weights.sum())
    if total <= 1e-8:
        return float(xs.mean()), float(ys.mean())
    return float((xs * weights).sum() / total), float((ys * weights).sum() / total)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _crop_prior(crop: Crop, saliency: np.ndarray, centroid: tuple[float, float] | None) -> float:
    """Cheap prior: keep salient content, avoid cutting it, favour thirds."""
    h, w = saliency.shape[:2]
    x1, y1, x2, y2 = crop.box
    region = saliency[y1:y2, x1:x2]
    mass = float(region.sum()) / (float(saliency.sum()) + 1e-8)

    # Saliency near the crop border indicates a cut subject.
    bw = max(2, int(round(crop.w * 0.06)))
    bh = max(2, int(round(crop.h * 0.06)))
    border = np.zeros_like(region, dtype=bool)
    border[:bh, :] = border[-bh:, :] = True
    border[:, :bw] = border[:, -bw:] = True
    border_mass = float(region[border].sum()) / (float(region.sum()) + 1e-8)

    thirds = 0.0
    if centroid is not None:
        cx, cy = centroid
        if x1 <= cx < x2 and y1 <= cy < y2:
            rx, ry = (cx - x1) / crop.w, (cy - y1) / crop.h
            d = min(
                abs(rx - tx) + abs(ry - ty)
                for tx, ty in product((1 / 3, 2 / 3), repeat=2)
            )
            thirds = float(np.exp(-(d / 0.22) ** 2))
        else:
            thirds = -0.5
    else:
        thirds = 0.5

    return 0.55 * mass + 0.30 * thirds + 0.15 * (1.0 - border_mass)


def _grid_positions(limit: int, size: int, steps: int) -> list[int]:
    if limit <= 0:
        return [0]
    if steps <= 1:
        return [limit // 2]
    return sorted({int(round(v)) for v in np.linspace(0, limit, steps)})


def generate_candidates(
    image: np.ndarray,
    saliency: np.ndarray,
    config: CandidateConfig | None = None,
) -> list[Crop]:
    cfg = config or CandidateConfig()
    h, w = image.shape[:2]
    centroid = saliency_centroid(saliency)

    proposals: list[Crop] = [Crop(0, 0, w, h)]  # always keep the original framing

    for aspect in cfg.aspect_ratios:
        max_w = min(w, int(round(h * aspect)))
        max_h = int(round(max_w / aspect))
        if max_w < cfg.min_side or max_h < cfg.min_side:
            continue

        for scale in cfg.scales:
            cw = int(round(max_w * scale))
            ch = int(round(cw / aspect))
            if cw < cfg.min_side or ch < cfg.min_side:
                continue
            cw, ch = min(cw, w), min(ch, h)

            xs = _grid_positions(w - cw, cw, cfg.grid_steps)
            ys = _grid_positions(h - ch, ch, cfg.grid_steps)
            for x, y in product(xs, ys):
                proposals.append(Crop(x, y, cw, ch))

            # Rule-of-thirds anchored proposals around the salient subject.
            if centroid is not None:
                cx, cy = centroid
                for tx, ty in product((1 / 3, 2 / 3), repeat=2):
                    x = int(round(cx - tx * cw))
                    y = int(round(cy - ty * ch))
                    x = int(_clamp(x, 0, w - cw))
                    y = int(_clamp(y, 0, h - ch))
                    proposals.append(Crop(x, y, cw, ch))

    # Greedy de-duplication, keeping the proposal with the better prior.
    scored = sorted(
        ((_crop_prior(c, saliency, centroid), c) for c in proposals),
        key=lambda item: item[0],
        reverse=True,
    )
    kept: list[Crop] = []
    for prior, crop in scored:
        if prior < 0:
            continue
        if all(crop.iou(other) < cfg.iou_threshold for other in kept):
            kept.append(crop)
        if len(kept) >= cfg.max_candidates:
            break
    return kept


def draw_candidates(
    image: np.ndarray,
    candidates: list[Crop],
    selected: Crop | None = None,
    max_draw: int = 12,
) -> np.ndarray:
    """Visualise candidate boxes (best-first ordering assumed)."""
    canvas = image.copy()
    for crop in candidates[:max_draw]:
        color = (80, 80, 80)
        thickness = 1
        if selected is not None and crop == selected:
            color, thickness = (0, 200, 0), 3
        cv2.rectangle(canvas, (crop.x, crop.y),
                      (crop.x + crop.w, crop.y + crop.h), color, thickness)
    if selected is not None:
        cv2.rectangle(canvas, (selected.x, selected.y),
                      (selected.x + selected.w, selected.y + selected.h),
                      (0, 200, 0), 3)
    return canvas
