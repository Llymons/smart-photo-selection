"""Optional multimodal-LLM (VLM) judge interface.

The mid-term prototype does not call any paid API.  Instead it defines a small
interface and a structured rubric, so a local model (e.g. Qwen2.5-VL) or a
hosted model can be plugged in later without touching the rest of the pipeline.

Typical integration::

    judge = CallableVLMRater(lambda image, context: my_vlm_fn(image, context))
    selector = SmartPhotoSelector(config, vlm_judge=judge)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class VLMAssessment:
    subject_integrity: float  # Is the main subject complete and unobstructed?
    composition: float        # Balance, placement, breathing room.
    moment: float             # Expression, gesture, timing ("decisive moment").
    aesthetic: float          # Colour, light, overall visual appeal.
    reasoning: str = ""

    @property
    def total(self) -> float:
        return float(
            0.30 * self.subject_integrity
            + 0.30 * self.composition
            + 0.20 * self.moment
            + 0.20 * self.aesthetic
        )


class VLMRater(Protocol):
    def assess(self, image: np.ndarray, context: str = "") -> VLMAssessment:
        ...  # pragma: no cover - protocol


class NeutralVLMRater:
    """Default no-op judge: returns a neutral score."""

    def assess(self, image: np.ndarray, context: str = "") -> VLMAssessment:
        return VLMAssessment(0.5, 0.5, 0.5, 0.5,
                             reasoning="neutral fallback (no VLM configured)")


class CallableVLMRater:
    """Adapt any callable returning a VLMAssessment or a dict."""

    def __init__(self, fn: Callable[[np.ndarray, str], VLMAssessment | dict]):
        self._fn = fn

    def assess(self, image: np.ndarray, context: str = "") -> VLMAssessment:
        result = self._fn(image, context)
        if isinstance(result, VLMAssessment):
            return result
        return VLMAssessment(
            subject_integrity=float(result.get("subject_integrity", 0.5)),
            composition=float(result.get("composition", 0.5)),
            moment=float(result.get("moment", 0.5)),
            aesthetic=float(result.get("aesthetic", 0.5)),
            reasoning=str(result.get("reasoning", "")),
        )


JUDGE_PROMPT = """你是一位经验丰富的摄影编辑，请从同一场景的一组照片中判断哪一张最值得保留。

请针对每一张候选照片，按以下维度分别打分（0-10 分，允许小数）：
1. 主体完整性：主体是否完整、清晰、无遮挡，是否被裁切；
2. 构图：主体位置（三分法/黄金分割）、画面平衡、留白、线条与地平线；
3. 瞬间：表情、姿态、动作是否自然，是否抓住了"决定性瞬间"；
4. 美学：光线、色彩、影调与整体视觉吸引力。

同时请用一句话说明理由，并特别注意：
- 不要因为照片"更亮"或"色彩更艳"就盲目给高分；
- 优先考虑主体与构图，其次才是色彩风格；
- 多张照片非常相似时，指出最关键的差异。

请只输出 JSON，格式如下：
{"subject_integrity": 0.0, "composition": 0.0, "moment": 0.0,
 "aesthetic": 0.0, "reasoning": "..."}
"""


def build_judge_prompt(context: str = "") -> str:
    """Return the rubric prompt, optionally with series-level context."""
    if context:
        return JUDGE_PROMPT + f"\n\n场景信息：{context}"
    return JUDGE_PROMPT
