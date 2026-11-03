"""Smart Photo Selection (SPS).

A lightweight, dependency-minimal prototype for selecting the most
aesthetically pleasing photo from a burst / series of near-duplicate photos,
with automatic candidate reframing (crop generation), multi-cue scoring and
within-series ranking.
"""

from .types import Crop, CandidateResult, PhotoResult, SeriesResult, ScoreBreakdown
from .pipeline import SmartPhotoSelector, SelectionConfig

__all__ = [
    "Crop",
    "CandidateResult",
    "PhotoResult",
    "SeriesResult",
    "ScoreBreakdown",
    "SmartPhotoSelector",
    "SelectionConfig",
]

__version__ = "0.1.0"
