"""Generate a synthetic photo series for smoke-testing and the demo.

The scene is a lakeside portrait at golden hour.  Six "shots" are rendered
from the same scene with different framing / degradations, which makes the
expected winner unambiguous while keeping the repository free of licensed
photos.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _gradient(h: int, w: int, top: tuple[int, int, int],
              bottom: tuple[int, int, int]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top_arr = np.array(top, dtype=np.float32)[None, None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, None, :]
    return np.broadcast_to(top_arr * (1 - t) + bottom_arr * t, (h, w, 3)).astype(np.uint8)


def _add_texture(image: np.ndarray, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5.0, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return cv2.GaussianBlur(noisy, (3, 3), 0.6)


def _vignette(image: np.ndarray, strength: float = 0.22) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    mask = 1.0 - strength * np.clip(dist / 1.414, 0, 1) ** 2
    return np.clip(image.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def make_scene(width: int = 1600, height: int = 1000, seed: int = 7) -> np.ndarray:
    horizon = int(height * 0.62)
    shore = int(height * 0.86)

    image = _gradient(height, width, (215, 165, 115), (245, 232, 205))
    # Sun and glow.
    sun = (1050, 235)
    glow = image.copy()
    cv2.circle(glow, sun, 110, (90, 215, 255), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 45)
    image = cv2.addWeighted(image, 0.75, glow, 0.55, 0)
    cv2.circle(image, sun, 52, (120, 235, 255), -1)

    # Distant mountain layers.
    rng = np.random.default_rng(seed)
    for layer, color in enumerate([(150, 150, 160), (120, 125, 135), (95, 105, 115)]):
        base = horizon - 40 + layer * 35
        pts = [(0, horizon)]
        x = 0
        while x <= width:
            y = base - rng.integers(40, 130) - 30 * np.sin(x / 240 + layer)
            pts.append((x, int(y)))
            x += 120
        pts += [(width, horizon)]
        cv2.fillPoly(image, [np.array(pts, dtype=np.int32)], color)

    # Lake with sun reflection.
    lake = _gradient(shore - horizon, width, (205, 180, 150), (120, 135, 140))
    image[horizon:shore] = lake
    for i in range(14):
        y = horizon + int((shore - horizon) * i / 14)
        alpha = 0.18 * (1 - i / 16)
        cv2.line(image, (sun[0] - 40 - i * 6, y), (sun[0] + 40 + i * 6, y),
                 (140, 225, 255), 2, cv2.LINE_AA)
        overlay = image.copy()
        cv2.line(overlay, (sun[0] - 40 - i * 6, y), (sun[0] + 40 + i * 6, y),
                 (150, 230, 255), 2)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Grass foreground with texture.
    grass = _gradient(height - shore, width, (70, 125, 85), (45, 95, 60))
    image[shore:] = grass

    # Trees on the left.
    for tx, ty, r in [(110, 480, 130), (230, 500, 100), (340, 520, 85), (70, 560, 90)]:
        cv2.rectangle(image, (tx - 10, ty), (tx + 10, shore + 25), (45, 60, 55), -1)
        cv2.circle(image, (tx, ty - 20), r, (45, 105, 70), -1)
        cv2.circle(image, (tx + 30, ty - 50), int(r * 0.8), (35, 90, 60), -1)

    # Person silhouette with a warm jacket (the salient subject).
    px, base_y = 985, shore + 15
    head_center = (px, base_y - 205)
    cv2.circle(image, head_center, 27, (60, 70, 95), -1)          # head
    cv2.rectangle(image, (px - 12, base_y - 180), (px + 12, base_y - 155),
                  (60, 70, 95), -1)                                 # neck
    body = np.array([[px - 45, base_y - 155], [px + 45, base_y - 155],
                     [px + 34, base_y - 60], [px - 34, base_y - 60]], np.int32)
    cv2.fillPoly(image, [body], (35, 75, 195))                     # red jacket
    cv2.fillPoly(image, [np.array([[px - 45, base_y - 155], [px - 30, base_y - 155],
                                   [px - 22, base_y - 70], [px - 34, base_y - 60]],
                                  np.int32)], (25, 60, 165))
    cv2.fillPoly(image, [np.array([[px - 34, base_y - 60], [px - 6, base_y - 60],
                                   [px - 9, base_y], [px - 30, base_y]], np.int32)],
                 (55, 55, 60))
    cv2.fillPoly(image, [np.array([[px + 6, base_y - 60], [px + 34, base_y - 60],
                                   [px + 30, base_y], [px + 9, base_y]], np.int32)],
                 (55, 55, 60))
    cv2.circle(image, (px - 38, base_y - 140), 10, (45, 85, 210), -1)  # arm
    cv2.circle(image, (px + 38, base_y - 140), 10, (45, 85, 210), -1)

    # Foreground flowers.
    for _ in range(90):
        fx = int(rng.integers(0, width))
        fy = int(rng.integers(shore + 20, height - 5))
        color = [(70, 90, 230), (120, 150, 250), (90, 210, 240), (200, 200, 240)][_ % 4]
        cv2.circle(image, (fx, fy), int(rng.integers(2, 6)), color, -1)

    image = _add_texture(image, seed)
    image = _vignette(image, 0.16)
    return image


def _crop_and_resize(scene: np.ndarray, box: tuple[int, int, int, int],
                     out_width: int = 1200) -> np.ndarray:
    x, y, w, h = box
    patch = scene[y:y + h, x:x + w]
    scale = out_width / patch.shape[1]
    return cv2.resize(patch, (out_width, int(round(patch.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def _degrade_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), 5.0)


def _degrade_exposure(image: np.ndarray) -> np.ndarray:
    bright = np.clip(image.astype(np.float32) + 85, 0, 255)
    mixed = cv2.addWeighted(bright, 0.85, np.full_like(bright, 255), 0.15, 0)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _degrade_dark_noisy(image: np.ndarray, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dark = image.astype(np.float32) * 0.34
    noise = rng.normal(0, 9.0, dark.shape)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)


def _degrade_hazy(image: np.ndarray) -> np.ndarray:
    haze = image.astype(np.float32) * 0.55 + 255 * 0.42
    return np.clip(haze, 0, 255).astype(np.uint8)


def generate_demo_series(out_dir: str | Path) -> list[Path]:
    """Create examples/demo_series/*.jpg and return sorted paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = make_scene()

    good_box = (580, 150, 1000, 750)  # 4:3, subject on the left third
    shots = {
        "01_balanced.jpg": _crop_and_resize(scene, good_box),
        "02_blurry.jpg": _degrade_blur(_crop_and_resize(scene, good_box)),
        "03_overexposed.jpg": _degrade_exposure(_crop_and_resize(scene, good_box)),
        "04_dark_noisy.jpg": _degrade_dark_noisy(_crop_and_resize(scene, good_box)),
        "05_hazy_low_contrast.jpg": _degrade_hazy(_crop_and_resize(scene, good_box)),
    }
    paths = []
    for name, image in shots.items():
        path = out_dir / name
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Failed to encode {name}")
        buf.tofile(str(path))
        paths.append(path)
    return paths


def main(out_dir: str | Path = "outputs/demo",
         series_dir: str | Path = "examples/demo_series") -> dict:
    from .pipeline import SmartPhotoSelector, save_results

    paths = generate_demo_series(series_dir)
    selector = SmartPhotoSelector()
    series = selector.select_series(paths)
    outputs = save_results(series, out_dir)

    print(f"Selected: {series.selected.path.name}")
    print(f"  series score: {series.selected.series_score:.4f}")
    for photo in series.ranking:
        print(f"  #{photo.rank} {photo.path.name:<30} "
              f"total={photo.best.score.total:.4f} "
              f"series={photo.series_score:.4f} crop={photo.best.crop.box}")
    print(f"Outputs written to {Path(out_dir).resolve()}")
    return {k: str(v) for k, v in outputs.items()}
