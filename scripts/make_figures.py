"""Generate the figures used in the design report."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sps.demo import make_scene  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1F2937"
BLUE = "#2F5597"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GRAY = "#6B7280"
LIGHT = "#EDF2F7"


def _setup_font() -> None:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            from matplotlib.font_manager import FontProperties
            font = FontProperties(fname=path)
            plt.rcParams["font.family"] = font.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def _box(ax, x, y, w, h, title, subtitle="", color=BLUE, fc=LIGHT):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                           linewidth=1.4, edgecolor=color, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
            fontsize=11.5, color=INK, fontweight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=8.2, color=GRAY, linespacing=1.45)


def _arrow(ax, x1, y1, x2, y2, color=GRAY, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.2, color=color,
        linestyle="--" if dashed else "-",
        connectionstyle="arc3,rad=0.0",
    ))


def fig_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y, h = 0.64, 0.26
    w = 0.155
    xs = [0.02, 0.205, 0.39, 0.575, 0.76]
    boxes = [
        ("输入组照", "同一场景 N 张\n（连拍 / 多机位）"),
        ("候选框生成", "多尺度 · 多比例\n三分法主体锚定 · 去重"),
        ("多线索评价", "技术质量 · 美学\n构图 · VLM 语义"),
        ("组内排序", "组内归一化\n加权融合 / Bradley–Terry"),
        ("输出与反馈", "最佳照片 + 取景框\n理由 · 用户反馈"),
    ]
    colors = [GRAY, TEAL, BLUE, BLUE, ORANGE]
    for x, (title, sub), color in zip(xs, boxes, colors):
        _box(ax, x, y, w, h, title, sub, color=color)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + w + 0.006, y + h / 2, xs[i + 1] - 0.006, y + h / 2)

    # Feedback loop.
    _arrow(ax, xs[4] + w / 2, y - 0.02, xs[3] + w / 2, y - 0.02, ORANGE, dashed=True)
    ax.text((xs[3] + xs[4] + w) / 2, y - 0.085, "用户反馈 / 主动学习", ha="center",
            fontsize=8, color=ORANGE)

    # Offline branch.
    _box(ax, 0.20, 0.14, 0.30, 0.20,
         "离线训练与校准",
         "Photo Triage 成对偏好 · AVA 美学\n自采组照 · 人工标注",
         color=BLUE, fc="#F7FAFC")
    _box(ax, 0.56, 0.14, 0.28, 0.20,
         "可插拔模型",
         "NIMA / MUSIQ / CLIP-Aesthetic\nVLM Judge（Top-k 复核）",
         color=BLUE, fc="#F7FAFC")
    _arrow(ax, 0.50, 0.24, 0.56, 0.24, BLUE)
    _arrow(ax, 0.70, 0.34, 0.66, y - 0.01, BLUE)

    fig.tight_layout()
    out = FIG_DIR / "fig1_pipeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _letterbox_bgr(image: np.ndarray, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """Fit an image into a fixed canvas with white padding, no distortion."""
    tw, th = size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def fig_candidates() -> Path:
    scene = make_scene()
    h, w = scene.shape[:2]
    boxes = [
        ((580, 150, 1000, 750), "A  4:3 · 主体位于左三分线", TEAL),
        ((700, 220, 750, 750), "B  1:1 · 紧凑构图", BLUE),
        ((420, 300, 1200, 675), "C  16:9 · 保留环境", ORANGE),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(_letterbox_bgr(scene), cv2.COLOR_BGR2RGB))
    # Map scene coordinates into the letterboxed canvas.
    th, tw = 480, 640
    scale = min(tw / w, th / h)
    ox, oy = (tw - int(w * scale)) / 2, (th - int(h * scale)) / 2
    for t in (1 / 3, 2 / 3):
        ax.axvline(ox + w * scale * t, color="white", lw=0.9, ls="--", alpha=0.9)
        ax.axhline(oy + h * scale * t, color="white", lw=0.9, ls="--", alpha=0.9)
    for (x, y, bw, bh), label, color in boxes:
        ax.add_patch(Rectangle((ox + x * scale, oy + y * scale),
                               bw * scale, bh * scale, fill=False,
                               edgecolor=color, lw=2.2))
        ax.text(ox + x * scale + 4, oy + y * scale + 18, label.split("  ")[0],
                color="white", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"))
    ax.set_title("(a) 原图、三分线与候选框", fontsize=10.5, loc="left")
    ax.axis("off")

    for i, ((x, y, bw, bh), label, color) in enumerate(boxes):
        ax = axes[(i + 1) // 2, (i + 1) % 2]
        crop = scene[y:y + bh, x:x + bw]
        ax.imshow(cv2.cvtColor(_letterbox_bgr(crop), cv2.COLOR_BGR2RGB))
        ax.set_title(f"({chr(98 + i)}) {label}", fontsize=10.5, loc="left")
        ax.axis("off")

    fig.tight_layout()
    out = FIG_DIR / "fig2_candidates.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_scores() -> Path:
    scores_path = ROOT / "outputs" / "demo" / "scores.json"
    if not scores_path.exists():
        from sps.demo import main as demo_main
        demo_main()
    data = json.loads(scores_path.read_text(encoding="utf-8"))
    ranking = data["ranking"]
    labels = []
    for item in ranking:
        stem = Path(item["path"]).stem
        labels.append(stem.split("_", 1)[1].replace("_", " ") if "_" in stem else stem)
    metrics = ["technical", "aesthetic", "composition", "total"]
    names = ["技术质量", "美学", "构图", "总分"]
    colors = [TEAL, "#8E7CC3", ORANGE, BLUE]

    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, (metric, name, color) in enumerate(zip(metrics, names, colors)):
        values = [item["best_score"][metric] for item in ranking]
        ax.bar(x + (i - 1.5) * width, values, width, label=name, color=color,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{item['rank']} {label}" for item, label in zip(ranking, labels)],
                       fontsize=9.5)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("得分（0–1）", fontsize=10)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = FIG_DIR / "fig3_scores.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_schedule() -> Path:
    tasks = [
        ("期中报告与仓库", date(2026, 11, 1), date(2026, 11, 3), BLUE),
        ("数据采集与标注", date(2026, 11, 4), date(2026, 11, 20), TEAL),
        ("基线与候选框生成", date(2026, 11, 18), date(2026, 12, 4), TEAL),
        ("评分模型与 LTR", date(2026, 12, 1), date(2026, 12, 18), ORANGE),
        ("VLM 集成与系统", date(2026, 12, 12), date(2026, 12, 30), ORANGE),
        ("实验、消融与用户研究", date(2026, 12, 25), date(2027, 1, 12), "#8E7CC3"),
        ("期末报告与演示", date(2027, 1, 8), date(2027, 1, 20), GRAY),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for i, (name, start, end, color) in enumerate(tasks):
        ax.barh(i, (end - start).days, left=mdates.date2num(start), height=0.56,
                color=color, alpha=0.9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=10)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(mdates.date2num(date(2026, 10, 28)), mdates.date2num(date(2027, 1, 24)))
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "fig4_schedule.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _setup_font()
    for fn in (fig_pipeline, fig_candidates, fig_scores, fig_schedule):
        print("wrote", fn())


if __name__ == "__main__":
    main()
