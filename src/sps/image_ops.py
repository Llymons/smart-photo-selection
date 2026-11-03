"""Low-level image utilities: robust IO, saliency, quality statistics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    """Read an image, supporting non-ASCII (e.g. Chinese) paths on Windows."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    """Write an image, supporting non-ASCII paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def saliency_map(image: np.ndarray) -> np.ndarray:
    """Spectral residual saliency (Hou & Zhang, 2007), returned in [0, 1].

    Implemented with NumPy so that opencv-contrib is not required.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32) / 255.0

    fft = np.fft.fft2(small)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    residual = np.exp(log_amp - cv2.blur(log_amp, (3, 3)))
    sal = np.abs(np.fft.ifft2(residual * np.exp(1j * phase))) ** 2
    sal = sal.astype(np.float32)
    sal = cv2.GaussianBlur(sal, (9, 9), 2.5)
    sal = cv2.normalize(sal, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian on a fixed-height copy (resolution robust)."""
    h, w = image.shape[:2]
    if h > 512:
        scale = 512 / h
        image = cv2.resize(image, (max(1, int(w * scale)), 512),
                           interpolation=cv2.INTER_AREA)
    gray = _to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(image: np.ndarray) -> tuple[float, float]:
    """Return (mean luminance in [0, 1], fraction of clipped pixels)."""
    gray = _to_gray(image).astype(np.float32) / 255.0
    clipped = float(np.mean((gray < 0.02) | (gray > 0.98)))
    return float(gray.mean()), clipped


def contrast(image: np.ndarray) -> float:
    gray = _to_gray(image).astype(np.float32) / 255.0
    return float(gray.std())


def colorfulness(image: np.ndarray) -> float:
    """Hasler & Suesstrunk colorfulness metric, roughly 0-100+."""
    b, g, r = cv2.split(image.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return std + 0.3 * mean


def noise_level(image: np.ndarray) -> float:
    """High-frequency residual after median filtering, in [0, 255] scale."""
    gray = _to_gray(image)
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    return float(np.mean(np.abs(residual)))


def face_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Optional Haar-cascade face detection (empty list if unavailable)."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = _to_gray(image)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(30, 30))
        return [tuple(int(v) for v in f) for f in faces]
    except Exception:
        return []
