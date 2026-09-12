"""Generate the synthetic legal-contract scans used by the Week 6 project.

The repository keeps the source text and labels under ``data/ground_truth`` and
regenerates scan images locally so large generated PNG files do not need to be
stored in Git history.
"""
from __future__ import annotations

from pathlib import Path
import textwrap
import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
GROUND_TRUTH = ROOT / "ground_truth"
SCAN_DIR = ROOT / "scans"
SEED = 20260912

ANGLES = [0.2, -0.4, 0.5, -0.7, 0.8, -0.9, 1.0, -1.2]
NOISE_SIGMA = [1.0, 1.5, 2.0, 2.5, 3.0, 9.0, 11.0, 13.0]
BLUR_SIGMA = [0.0, 0.0, 0.0, 0.3, 0.4, 0.8, 1.0, 1.2]


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_source(text: str, width: int = 86) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.rstrip()
        if not raw:
            lines.append("")
        elif raw.isupper() and len(raw) < 40:
            lines.append(raw)
        else:
            lines.extend(textwrap.wrap(raw, width=width, break_long_words=False))
    return lines


def render_contract(text: str, sample_number: int) -> np.ndarray:
    width, height = 1700, 2200
    margin_x, margin_y = 125, 115
    font = load_font(27)
    heading_font = load_font(28)
    line_height = 42

    page = Image.new("L", (width, height), color=248)
    draw = ImageDraw.Draw(page)
    y = margin_y

    for line in wrap_source(text):
        if y > height - 170:
            break
        active_font = heading_font if line.isupper() and line else font
        draw.text((margin_x, y), line, fill=30, font=active_font)
        y += line_height

    footer = f"Synthetic contract sample {sample_number} — Week 6 safe dataset"
    draw.text((margin_x, height - 100), footer, fill=115, font=load_font(20))
    image = np.array(page)

    # Add a mild left-to-right illumination gradient. Later samples are harder.
    strength = 4 + sample_number * 1.5
    gradient = np.linspace(0, strength, width, dtype=np.float32)
    image = np.clip(image.astype(np.float32) - gradient[None, :], 0, 255)

    sigma = BLUR_SIGMA[sample_number - 1]
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    rng = np.random.default_rng(SEED + sample_number)
    noise = rng.normal(0, NOISE_SIGMA[sample_number - 1], image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    angle = ANGLES[sample_number - 1]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return image


def main() -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = GROUND_TRUTH / "labels.csv"
    if not labels_path.exists():
        raise SystemExit("data/ground_truth/labels.csv was not found.")

    with labels_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("No synthetic contracts were found in labels.csv.")

    for index, row in enumerate(rows, start=1):
        image = render_contract(row["text"], index)
        output = SCAN_DIR / f"contract_{index:02d}.png"
        cv2.imwrite(str(output), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(f"generated {output.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
