"""End-to-end smart photo selection pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import image_ops
from .candidates import CandidateConfig, draw_candidates, generate_candidates
from .ranking import PairwiseRanker, rank_photos
from .scoring import AestheticScorer, HeuristicAestheticScorer, ScoringWeights, score_crop
from .types import CandidateResult, Crop, PhotoResult, SeriesResult
from .vlm import NeutralVLMRater, VLMAssessment, VLMRater


@dataclass
class SelectionConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    process_max_side: int = 1600
    vlm_top_k: int = 5
    context: str = ""


class SmartPhotoSelector:
    """Select the best photo (and its best framing) from a series."""

    def __init__(
        self,
        config: SelectionConfig | None = None,
        aesthetic_scorer: AestheticScorer | None = None,
        vlm_judge: VLMRater | None = None,
        ranker: PairwiseRanker | None = None,
    ):
        self.config = config or SelectionConfig()
        self.aesthetic_scorer = aesthetic_scorer or HeuristicAestheticScorer()
        self.vlm_judge = vlm_judge or NeutralVLMRater()
        self.ranker = ranker

    # ------------------------------------------------------------------ #
    # Single photo
    # ------------------------------------------------------------------ #
    def select_photo(self, path: str | Path) -> PhotoResult:
        path = Path(path)
        original = image_ops.read_image(path)
        original_h, original_w = original.shape[:2]
        analysis = image_ops.resize_max(original, self.config.process_max_side)
        ah, aw = analysis.shape[:2]
        scale = aw / original_w

        saliency = image_ops.saliency_map(analysis)
        faces = image_ops.face_boxes(analysis)
        crops = generate_candidates(analysis, saliency, self.config.candidate)

        results = []
        for crop in crops:
            score = score_crop(
                analysis, crop, saliency, self.config.weights,
                aesthetic_scorer=self.aesthetic_scorer,
                semantic_score=0.5,
                face_boxes=faces,
            )
            results.append((crop, score))

        # Run the (expensive) VLM only on the most promising framings.
        if not isinstance(self.vlm_judge, NeutralVLMRater):
            results.sort(key=lambda item: item[1].total, reverse=True)
            judged = []
            for crop, score in results[: self.config.vlm_top_k]:
                x1, y1, x2, y2 = crop.box
                assessment = self.vlm_judge.assess(
                    analysis[y1:y2, x1:x2], self.config.context
                )
                score.semantic = float(assessment.total)
                w = self.config.weights.normalised()
                score.total = (
                    w.technical * score.technical
                    + w.aesthetic * score.aesthetic
                    + w.composition * score.composition
                    + w.semantic * score.semantic
                )
                score.details["vlm"] = {
                    "subject_integrity": assessment.subject_integrity,
                    "composition": assessment.composition,
                    "moment": assessment.moment,
                    "aesthetic": assessment.aesthetic,
                    "reasoning": assessment.reasoning,
                }
                judged.append((crop, score))
            rest = results[self.config.vlm_top_k:]
            results = sorted(judged + rest, key=lambda item: item[1].total, reverse=True)

        best_crop, best_score = max(results, key=lambda item: item[1].total)

        # Map the analysis-space crop back to the original resolution.
        ox = min(max(0, int(round(best_crop.x / scale))), original_w - 1)
        oy = min(max(0, int(round(best_crop.y / scale))), original_h - 1)
        original_crop = Crop(
            x=ox,
            y=oy,
            w=max(1, min(original_w - ox, int(round(best_crop.w / scale)))),
            h=max(1, min(original_h - oy, int(round(best_crop.h / scale)))),
        )
        candidates = [
            CandidateResult(crop=c, score=s) for c, s in results
        ]
        return PhotoResult(
            path=path,
            image_size=(original_w, original_h),
            best=CandidateResult(crop=original_crop, score=best_score),
            candidates=candidates,
        )

    # ------------------------------------------------------------------ #
    # Series
    # ------------------------------------------------------------------ #
    def select_series(self, paths: list[str | Path]) -> SeriesResult:
        if not paths:
            raise ValueError("A photo series needs at least one image")
        photos = [self.select_photo(p) for p in paths]
        ranked = rank_photos(photos, self.config.weights, self.ranker)
        return SeriesResult(photos=photos, selected=ranked[0], ranking=ranked)


# ---------------------------------------------------------------------- #
# Output helpers
# ---------------------------------------------------------------------- #
def _scaled_box(crop: Crop, sx: float, sy: float) -> tuple[int, int, int, int]:
    return (int(round(crop.x * sx)), int(round(crop.y * sy)),
            int(round((crop.x + crop.w) * sx)),
            int(round((crop.y + crop.h) * sy)))


def save_selected_crop(series: SeriesResult, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = image_ops.read_image(series.selected.path)
    crop = series.selected.best.crop
    x1, y1, x2, y2 = crop.box
    patch = image[max(0, y1):y2, max(0, x1):x2]
    out_path = out_dir / "selected.jpg"
    image_ops.write_image(out_path, patch)
    return out_path


def save_candidates_visual(series: SeriesResult, out_dir: str | Path,
                           max_draw: int = 12) -> Path:
    out_dir = Path(out_dir)
    image = image_ops.read_image(series.selected.path)
    analysis = image_ops.resize_max(image, 1600)
    sx = analysis.shape[1] / image.shape[1]
    sy = analysis.shape[0] / image.shape[0]
    crop = series.selected.best.crop
    selected_in_analysis = Crop(int(round(crop.x * sx)), int(round(crop.y * sy)),
                                max(1, int(round(crop.w * sx))),
                                max(1, int(round(crop.h * sy))))
    # Regenerate candidate boxes in the analysis space for a faithful drawing.
    saliency = image_ops.saliency_map(analysis)
    candidates = generate_candidates(analysis, saliency)
    canvas = draw_candidates(analysis, candidates, selected_in_analysis, max_draw)
    out_path = out_dir / "candidates.jpg"
    image_ops.write_image(out_path, canvas)
    return out_path


def make_contact_sheet(series: SeriesResult, out_dir: str | Path,
                       thumb_height: int = 320) -> Path:
    out_dir = Path(out_dir)
    thumbs = []
    for photo in series.photos:
        image = image_ops.read_image(photo.path)
        scale = thumb_height / image.shape[0]
        thumb = cv2.resize(image, (max(1, int(image.shape[1] * scale)), thumb_height))
        x1, y1, x2, y2 = _scaled_box(photo.best.crop, scale, scale)
        color = (0, 200, 0) if photo is series.selected else (150, 150, 150)
        thickness = 3 if photo is series.selected else 1
        cv2.rectangle(thumb, (x1, y1), (x2, y2), color, thickness)
        label = f"#{photo.rank} {photo.path.name} {photo.series_score:.3f}"
        cv2.rectangle(thumb, (0, 0), (thumb.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(thumb, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    gap = 8
    width = sum(t.shape[1] for t in thumbs) + gap * (len(thumbs) + 1)
    canvas = np.full((thumb_height + 2 * gap, width, 3), 30, dtype=np.uint8)
    x = gap
    for thumb in thumbs:
        canvas[gap:gap + thumb_height, x:x + thumb.shape[1]] = thumb
        x += thumb.shape[1] + gap
    out_path = out_dir / "contact_sheet.jpg"
    image_ops.write_image(out_path, canvas, quality=92)
    return out_path


def save_results(series: SeriesResult, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected": save_selected_crop(series, out_dir),
        "contact_sheet": make_contact_sheet(series, out_dir),
        "candidates": save_candidates_visual(series, out_dir),
    }
    report_path = out_dir / "scores.json"
    report_path.write_text(
        json.dumps(series.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["scores"] = report_path
    return paths
