"""Smoke tests for candidate generation, scoring, ranking and the demo."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sps.candidates import CandidateConfig, generate_candidates  # noqa: E402
from sps.image_ops import saliency_map  # noqa: E402
from sps.pipeline import SmartPhotoSelector  # noqa: E402
from sps.ranking import PairwiseRanker, rank_photos  # noqa: E402
from sps.scoring import ScoringWeights  # noqa: E402
from sps.types import CandidateResult, Crop, PhotoResult, ScoreBreakdown  # noqa: E402


def _dummy_photo(name: str, base: float) -> PhotoResult:
    score = ScoreBreakdown(base, base, base, base, base)
    return PhotoResult(
        path=Path(name),
        image_size=(100, 100),
        best=CandidateResult(Crop(0, 0, 100, 100), score),
        candidates=[],
    )


class TestGeometry(unittest.TestCase):
    def test_iou(self):
        a = Crop(0, 0, 100, 100)
        self.assertAlmostEqual(a.iou(a), 1.0)
        b = Crop(50, 50, 100, 100)
        self.assertAlmostEqual(a.iou(b), 2500 / 17500)
        self.assertEqual(a.iou(Crop(200, 200, 10, 10)), 0.0)


class TestCandidates(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        self.saliency = saliency_map(self.image)

    def test_bounds_and_dedup(self):
        config = CandidateConfig(max_candidates=40, iou_threshold=0.9)
        crops = generate_candidates(self.image, self.saliency, config)
        self.assertTrue(crops)
        self.assertLessEqual(len(crops), config.max_candidates)
        for crop in crops:
            self.assertGreaterEqual(crop.x, 0)
            self.assertGreaterEqual(crop.y, 0)
            self.assertLessEqual(crop.x + crop.w, 640)
            self.assertLessEqual(crop.y + crop.h, 480)
            self.assertGreater(crop.w, 0)
            self.assertGreater(crop.h, 0)
        for i, a in enumerate(crops):
            for b in crops[i + 1:]:
                self.assertLessEqual(a.iou(b), 0.9 + 1e-9)

    def test_full_frame_included(self):
        crops = generate_candidates(self.image, self.saliency, CandidateConfig())
        self.assertTrue(any(c.box == (0, 0, 640, 480) for c in crops))


class TestRanking(unittest.TestCase):
    def test_order_and_rank(self):
        photos = [_dummy_photo(f"{i}.jpg", s) for i, s in enumerate([0.2, 0.8, 0.5])]
        ranked = rank_photos(photos, ScoringWeights())
        self.assertEqual(ranked[0].path.name, "1.jpg")
        self.assertEqual([p.rank for p in ranked], [1, 2, 3])

    def test_pairwise_ranker_separates(self):
        preferred = np.array([[0.8, 0.8, 0.8, 0.8], [0.7, 0.7, 0.7, 0.7]])
        other = np.array([[0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]])
        ranker = PairwiseRanker(epochs=400).fit(preferred, other)
        self.assertTrue(np.all(ranker.predict(preferred) > ranker.predict(other)))


class TestDemo(unittest.TestCase):
    def test_demo_selects_balanced_photo(self):
        from sps.demo import generate_demo_series

        with tempfile.TemporaryDirectory() as tmp:
            series_dir = Path(tmp) / "series"
            paths = generate_demo_series(series_dir)
            selector = SmartPhotoSelector()
            result = selector.select_series(paths)
            self.assertEqual(result.selected.path.name, "01_balanced.jpg")


if __name__ == "__main__":
    unittest.main()
