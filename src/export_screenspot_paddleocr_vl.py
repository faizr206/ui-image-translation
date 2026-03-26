#!/usr/bin/env python3
"""Sample ScreenSpot images and export PaddleOCR-VL-1.5 spotting results.

Example:
    uv run python src/export_screenspot_paddleocr_vl.py \
    --num-samples 10 \
    --target-lang id \
    --source-lang auto \
    --inpaint-backend lama \
    --inpaint-model-id stable-diffusion-v1-5/stable-diffusion-inpainting \
    --output-dir outputs/screenspot_vl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

# Keep PaddleX fully local-friendly by skipping remote hoster connectivity checks.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
from datasets import load_dataset
from paddleocr import PaddleOCRVL
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample N images from rootsautomation/ScreenSpot, run local PaddleOCR-VL-1.5 "
            "spotting OCR, optional translation, and optional text-removal inpainting."
        )
    )
    parser.add_argument("--num-samples", "-n", type=int, required=True, help="Number of samples to process.")
    parser.add_argument("--dataset", type=str, default="rootsautomation/ScreenSpot", help="Hugging Face dataset id.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to sample from.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/screenspot_vl"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling.")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Use first N rows instead of random sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu:0",
        help="PaddleX device (e.g. cpu, gpu:0). If omitted, Paddle default is used.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="max_new_tokens passed to PaddleOCRVL.predict().",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="id",
        help="Target language code for Argos Translate (e.g. en, id, ar). If omitted, translation is skipped.",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="en",
        help="Source language code for Argos Translate (default: auto).",
    )
    parser.add_argument(
        "--disable-inpainting",
        action="store_true",
        help="Disable text-removal inpainting stage.",
    )
    parser.add_argument(
        "--inpaint-backend",
        type=str,
        default="lama",
        choices=("sd", "lama"),
        help=(
            "Inpainting backend. `sd` uses Stable Diffusion tiled inpainting; "
            "`lama` uses full-image LaMa inpainting."
        ),
    )
    parser.add_argument(
        "--inpaint-model-id",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
        help="Hugging Face model id for Stable Diffusion inpainting (used when --inpaint-backend sd).",
    )
    parser.add_argument(
        "--inpaint-prompt",
        type=str,
        default="Clean UI background, remove all text, keep layout and visual style.",
        help="Prompt used by Stable Diffusion inpainting.",
    )
    parser.add_argument(
        "--inpaint-negative-prompt",
        type=str,
        default="text, letters, words, watermark, logo, artifacts, blurry",
        help="Negative prompt used by Stable Diffusion inpainting.",
    )
    parser.add_argument(
        "--inpaint-steps",
        type=int,
        default=30,
        help="Number of diffusion steps for inpainting.",
    )
    parser.add_argument(
        "--inpaint-guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for inpainting.",
    )
    parser.add_argument(
        "--inpaint-strength",
        type=float,
        default=0.99,
        help="Inpainting strength (0 to 1).",
    )
    parser.add_argument(
        "--mask-padding",
        type=int,
        default=6,
        help="Padding (in pixels) to expand OCR text mask before inpainting.",
    )
    parser.add_argument(
        "--inpaint-tile-size",
        type=int,
        default=512,
        help="Tile size for inpainting when using Stable Diffusion backend.",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=Path("assets/fonts"),
        help="Directory containing candidate font files (searched recursively in subfolders).",
    )
    parser.add_argument(
        "--font-match-max-fonts",
        type=int,
        default=200,
        help="Maximum number of font files to evaluate for SSIM font matching.",
    )
    return parser.parse_args()


def normalize_polygon(poly: Any) -> list[tuple[float, float]]:
    if poly is None:
        return []

    if hasattr(poly, "tolist"):
        poly = poly.tolist()

    if not isinstance(poly, (list, tuple)) or len(poly) == 0:
        return []

    if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
        x1, y1, x2, y2 = map(float, poly)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    points: list[tuple[float, float]] = []
    for p in poly:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            points.append((float(p[0]), float(p[1])))
    return points


def denormalize_bbox_if_needed(bbox: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return [x1, y1, x2, y2]


def build_ocr_model(args: argparse.Namespace) -> PaddleOCRVL:
    model_kwargs: dict[str, Any] = {
        "pipeline_version": "v1.5",
        "use_layout_detection": False,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
    }
    if args.device:
        model_kwargs["device"] = args.device

    return PaddleOCRVL(**model_kwargs)


def safe_text(text_item: Any) -> str:
    if isinstance(text_item, tuple) and len(text_item) > 0:
        return str(text_item[0])
    return str(text_item)


def get_installed_argos_pairs(argos_translate: Any) -> set[tuple[str, str]]:
    installed_languages = argos_translate.get_installed_languages()
    available_pairs: set[tuple[str, str]] = set()
    for language in installed_languages:
        for translation in getattr(language, "translations_from", []):
            from_code = str(getattr(getattr(translation, "from_lang", None), "code", "")).lower().strip()
            to_code = str(getattr(getattr(translation, "to_lang", None), "code", "")).lower().strip()
            if from_code and to_code:
                available_pairs.add((from_code, to_code))
    return available_pairs


def try_install_argos_package(argos_package: Any, from_code: str, to_code: str) -> bool:
    try:
        print(f"[info] Trying to download Argos model: {from_code} -> {to_code}")
        argos_package.update_package_index()
        available_packages = argos_package.get_available_packages()
        package_to_install = next(
            (
                pkg
                for pkg in available_packages
                if str(getattr(pkg, "from_code", "")).lower() == from_code
                and str(getattr(pkg, "to_code", "")).lower() == to_code
            ),
            None,
        )
        if package_to_install is None:
            print(f"[warn] No downloadable Argos package found for {from_code} -> {to_code}")
            return False

        download_path = package_to_install.download()
        argos_package.install_from_path(download_path)
        print(f"[info] Installed Argos model: {from_code} -> {to_code}")
        return True
    except Exception as exc:
        print(f"[warn] Failed to auto-install Argos model {from_code}->{to_code}: {exc}")
        return False


def build_translator(source_lang: str, target_lang: str | None):
    if not target_lang:
        return None

    try:
        import argostranslate.package as argos_package
        import argostranslate.translate as argos_translate
    except ImportError as exc:
        raise ImportError(
            "argostranslate is not installed. Run `uv sync` (or `uv add argostranslate`) and retry."
        ) from exc

    source_lang = (source_lang or "auto").strip().lower()
    target_lang = target_lang.strip().lower()
    available_pairs = get_installed_argos_pairs(argos_translate)

    if source_lang == "auto":
        candidates = sorted({from_code for from_code, to_code in available_pairs if to_code == target_lang})
        if not candidates:
            # Prefer en->target for auto mode if no model is installed yet.
            if try_install_argos_package(argos_package, "en", target_lang):
                available_pairs = get_installed_argos_pairs(argos_translate)
                candidates = sorted({from_code for from_code, to_code in available_pairs if to_code == target_lang})
        if not candidates:
            raise ValueError(f"No installed Argos model can translate to '{target_lang}'.")
        resolved_source = "en" if "en" in candidates else candidates[0]
        if len(candidates) > 1:
            print(
                f"[warn] --source-lang auto is ambiguous for target '{target_lang}'. "
                f"Using '{resolved_source}'. Candidates: {', '.join(candidates)}"
            )
    else:
        resolved_source = source_lang

    if (resolved_source, target_lang) not in available_pairs and resolved_source != target_lang:
        if try_install_argos_package(argos_package, resolved_source, target_lang):
            available_pairs = get_installed_argos_pairs(argos_translate)
    if (resolved_source, target_lang) not in available_pairs and resolved_source != target_lang:
        raise ValueError(f"No installed Argos translation model for '{resolved_source}' -> '{target_lang}'.")

    return {
        "module": argos_translate,
        "source_lang": resolved_source,
        "target_lang": target_lang,
    }


def translate_texts(
    texts: list[str],
    translator: Any,
) -> list[str]:
    if not texts or translator is None:
        return texts

    translator_module = translator["module"]
    source_lang = translator["source_lang"]
    target_lang = translator["target_lang"]

    out: list[str] = []
    for text in texts:
        try:
            if source_lang == target_lang:
                out.append(text)
            else:
                out.append(str(translator_module.translate(text, source_lang, target_lang)))
        except Exception:
            out.append(text)
    return out


def discover_font_files(font_dir: Path, max_fonts: int) -> list[Path]:
    if max_fonts <= 0:
        return []
    if not font_dir.exists():
        return []

    # Recursive search so nested directories like assets/fonts/<family>/* are included.
    exts = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}
    files = [p for p in sorted(font_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    return files[:max_fonts]


def to_hex_rgb(values: np.ndarray) -> str:
    r, g, b = (int(np.clip(v, 0, 255)) for v in values[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def choose_text_mask(
    roi_mask: np.ndarray,
    bright_candidate: np.ndarray,
    dark_candidate: np.ndarray,
) -> np.ndarray:
    roi_pixels = int(roi_mask.sum())
    if roi_pixels <= 0:
        return np.zeros_like(roi_mask, dtype=bool)

    candidates: list[tuple[int, np.ndarray]] = []
    for candidate in (bright_candidate, dark_candidate):
        count = int(candidate.sum())
        if count <= 0:
            continue
        ratio = count / float(roi_pixels)
        if 0.01 <= ratio <= 0.95:
            candidates.append((count, candidate))

    if candidates:
        # Text strokes usually occupy less area than the surrounding box.
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    bright_count = int(bright_candidate.sum())
    dark_count = int(dark_candidate.sum())
    if bright_count <= 0 and dark_count <= 0:
        return np.zeros_like(roi_mask, dtype=bool)
    return bright_candidate if bright_count <= dark_count else dark_candidate


def extract_region_and_text_mask(
    image: Image.Image,
    polygon: list[list[float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(polygon) < 3:
        return None

    width, height = image.size
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    left = max(0, int(np.floor(min(xs))))
    top = max(0, int(np.floor(min(ys))))
    right = min(width, int(np.ceil(max(xs))))
    bottom = min(height, int(np.ceil(max(ys))))
    if right <= left or bottom <= top:
        return None

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    crop_rgb = np.array(crop, dtype=np.uint8)
    crop_h, crop_w = crop_rgb.shape[:2]
    if crop_h < 2 or crop_w < 2:
        return None

    local_poly = np.array(
        [[float(x) - left, float(y) - top] for x, y in polygon],
        dtype=np.float32,
    )
    local_poly[:, 0] = np.clip(local_poly[:, 0], 0, crop_w - 1)
    local_poly[:, 1] = np.clip(local_poly[:, 1], 0, crop_h - 1)
    local_poly_i = np.round(local_poly).astype(np.int32)

    roi_mask_u8 = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask_u8, [local_poly_i], 255)
    roi_mask = roi_mask_u8 > 0

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_inv = cv2.bitwise_not(threshold)

    bright_candidate = (threshold > 0) & roi_mask
    dark_candidate = (threshold_inv > 0) & roi_mask
    text_mask = choose_text_mask(roi_mask, bright_candidate, dark_candidate)
    if int(text_mask.sum()) <= 0:
        return None

    return crop_rgb, text_mask


def compute_text_color(crop_rgb: np.ndarray, text_mask: np.ndarray) -> dict[str, Any] | None:
    if crop_rgb.size == 0 or int(text_mask.sum()) <= 0:
        return None

    pixels = crop_rgb[text_mask]
    if pixels.size == 0:
        return None

    median_rgb = np.median(pixels, axis=0)
    mean_rgb = np.mean(pixels, axis=0)

    median_rgb_i = [int(round(v)) for v in median_rgb.tolist()]
    mean_rgb_i = [int(round(v)) for v in mean_rgb.tolist()]

    return {
        "pixel_count": int(pixels.shape[0]),
        "median_rgb": median_rgb_i,
        "median_hex": to_hex_rgb(np.array(median_rgb_i, dtype=np.float64)),
        "mean_rgb": mean_rgb_i,
        "mean_hex": to_hex_rgb(np.array(mean_rgb_i, dtype=np.float64)),
    }


def compute_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    can = candidate.astype(np.float64)

    if ref.shape != can.shape:
        can = cv2.resize(can, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_ref = float(ref.mean())
    mu_can = float(can.mean())
    var_ref = float(ref.var())
    var_can = float(can.var())
    cov = float(((ref - mu_ref) * (can - mu_can)).mean())

    numerator = (2 * mu_ref * mu_can + c1) * (2 * cov + c2)
    denominator = (mu_ref**2 + mu_can**2 + c1) * (var_ref + var_can + c2)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def best_font_ssim_for_text(text: str, target_mask: np.ndarray, font_path: Path) -> float:
    text = text.strip()
    if not text:
        return -1.0

    h, w = target_mask.shape
    if h < 2 or w < 2:
        return -1.0

    target_img = (target_mask.astype(np.uint8) * 255).astype(np.uint8)
    size_candidates = sorted(
        set(max(8, int(h * ratio)) for ratio in (0.6, 0.8, 1.0, 1.2, 1.4))
    )

    best_score = -1.0
    for font_size in size_candidates:
        try:
            font = ImageFont.truetype(str(font_path), size=font_size)
        except Exception:
            continue

        probe = Image.new("L", (w, h), 0)
        probe_draw = ImageDraw.Draw(probe)
        bbox = probe_draw.textbbox((0, 0), text, font=font)
        if not bbox:
            continue

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= 0 or text_h <= 0:
            continue

        x_positions = sorted(
            set(
                [
                    -bbox[0],
                    max(0, (w - text_w) // 2) - bbox[0],
                    max(0, w - text_w) - bbox[0],
                ]
            )
        )
        y_positions = sorted(
            set(
                [
                    -bbox[1],
                    max(0, (h - text_h) // 2) - bbox[1],
                    max(0, h - text_h) - bbox[1],
                ]
            )
        )

        for x_pos in x_positions:
            for y_pos in y_positions:
                canvas = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(canvas)
                draw.text((x_pos, y_pos), text, font=font, fill=255)
                rendered = np.array(canvas, dtype=np.uint8)
                score = compute_ssim(target_img, rendered)
                if score > best_score:
                    best_score = score

    return best_score


def match_font_family(
    text: str,
    text_mask: np.ndarray,
    font_files: list[Path],
) -> dict[str, Any] | None:
    if not text.strip() or not font_files or int(text_mask.sum()) <= 0:
        return None

    best_font: Path | None = None
    best_score = -1.0

    for font_path in font_files:
        score = best_font_ssim_for_text(text, text_mask, font_path)
        if score > best_score:
            best_score = score
            best_font = font_path

    if best_font is None:
        return None

    return {
        "font_family": best_font.stem,
        "font_file": best_font.name,
        "font_path": str(best_font),
        "ssim": round(float(best_score), 6),
    }


def extract_text_style(
    image: Image.Image,
    text: str,
    polygon: list[list[float]],
    font_files: list[Path],
) -> dict[str, Any]:
    region = extract_region_and_text_mask(image, polygon)
    if region is None:
        return {
            "text_color": None,
            "font_match": None,
        }

    crop_rgb, text_mask = region
    color = compute_text_color(crop_rgb, text_mask)
    font_match = match_font_family(text, text_mask, font_files)
    return {
        "text_color": color,
        "font_match": font_match,
    }


def build_inpaint_pipeline(args: argparse.Namespace) -> Any:
    if args.disable_inpainting:
        return None

    if args.inpaint_backend == "lama":
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:
            raise ImportError(
                "LaMa backend requires `simple-lama-inpainting`. "
                "Install it with `uv add simple-lama-inpainting` and retry."
            ) from exc

        return {
            "backend": "lama",
            "pipeline": SimpleLama(),
        }

    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError as exc:
        raise ImportError(
            "Stable Diffusion inpainting requires diffusers + torch. Run `uv sync` and retry."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.inpaint_model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return {
        "backend": "sd",
        "pipeline": pipe,
    }


def build_text_mask(
    image_size: tuple[int, int],
    ocr_items: list[dict[str, Any]],
    mask_padding: int,
) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    for item in ocr_items:
        polygon = item.get("polygon", [])
        if not polygon:
            continue

        points = [(float(x), float(y)) for x, y in polygon]
        draw.polygon(points, fill=255)

    if mask_padding > 0:
        # Expand white regions so full glyph strokes are covered.
        size = max(3, mask_padding * 2 + 1)
        if size % 2 == 0:
            size += 1
        mask = mask.filter(ImageFilter.MaxFilter(size=size))
    return mask


def inpaint_text_regions(
    image: Image.Image,
    mask: Image.Image,
    inpaint_pipe: Any,
    args: argparse.Namespace,
) -> Image.Image:
    if inpaint_pipe is None or mask.getbbox() is None:
        return image.copy()

    backend = inpaint_pipe.get("backend", "sd")
    pipeline = inpaint_pipe.get("pipeline")

    if backend == "lama":
        try:
            # LaMa runs on the entire image directly (no chunk/tile splitting).
            result = pipeline(image, mask.convert("L"))
            if isinstance(result, Image.Image):
                return result.convert("RGB")
            return Image.fromarray(result).convert("RGB")
        except Exception as exc:
            print(f"[warn] LaMa inpainting failed: {exc}. Keeping original image.")
            return image.copy()

    tile_size = max(64, int(args.inpaint_tile_size))
    width, height = image.size
    stitched = image.copy()

    try:
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)
                crop_box = (left, top, right, bottom)

                tile_mask = mask.crop(crop_box)
                if tile_mask.getbbox() is None:
                    continue

                tile_img = image.crop(crop_box)
                tile_w, tile_h = tile_img.size

                # Diffusion inpainting expects fixed-size tiles, so pad edge tiles.
                tile_img_pad = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                tile_img_pad.paste(tile_img, (0, 0))
                tile_mask_pad = Image.new("L", (tile_size, tile_size), 0)
                tile_mask_pad.paste(tile_mask, (0, 0))

                result = pipeline(
                    prompt=args.inpaint_prompt,
                    negative_prompt=args.inpaint_negative_prompt,
                    image=tile_img_pad,
                    mask_image=tile_mask_pad,
                    num_inference_steps=args.inpaint_steps,
                    guidance_scale=args.inpaint_guidance_scale,
                    strength=args.inpaint_strength,
                )
                tile_inpainted_pad = result.images[0].convert("RGB")
                tile_inpainted = tile_inpainted_pad.crop((0, 0, tile_w, tile_h))

                # Keep only masked pixels from inpainting and paste back into full image.
                tile_merged = Image.composite(tile_inpainted, tile_img, tile_mask)
                stitched.paste(tile_merged, (left, top))

        return stitched
    except Exception as exc:
        print(f"[warn] Inpainting failed: {exc}. Keeping original image.")
        return image.copy()


def polygon_to_bbox(
    polygon: list[list[float]],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    if len(polygon) < 3:
        return None

    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    left = max(0, int(np.floor(min(xs))))
    top = max(0, int(np.floor(min(ys))))
    right = min(image_width, int(np.ceil(max(xs))))
    bottom = min(image_height, int(np.ceil(max(ys))))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def resolve_render_color(item: dict[str, Any]) -> tuple[int, int, int]:
    text_color = item.get("text_color")
    if isinstance(text_color, dict):
        for key in ("median_rgb", "mean_rgb"):
            rgb = text_color.get(key)
            if isinstance(rgb, list) and len(rgb) >= 3:
                return (
                    int(np.clip(rgb[0], 0, 255)),
                    int(np.clip(rgb[1], 0, 255)),
                    int(np.clip(rgb[2], 0, 255)),
                )
    return (0, 0, 0)


def resolve_render_font_path(item: dict[str, Any], font_files: list[Path]) -> Path | None:
    font_match = item.get("font_match")
    if not isinstance(font_match, dict):
        return None

    matched_path = font_match.get("font_path")
    if isinstance(matched_path, str) and matched_path.strip():
        path = Path(matched_path)
        if path.exists():
            return path

    matched_file = font_match.get("font_file")
    if isinstance(matched_file, str) and matched_file.strip():
        for font_path in font_files:
            if font_path.name == matched_file:
                return font_path

    return None


def load_font_for_size(font_path: Path | None, font_size: int) -> ImageFont.ImageFont:
    tried: set[str] = set()
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=font_size)
        except Exception:
            tried.add(str(font_path))

    # Pillow commonly bundles/supports this fallback.
    fallback = "DejaVuSans.ttf"
    if fallback not in tried:
        try:
            return ImageFont.truetype(fallback, size=font_size)
        except Exception:
            pass

    return ImageFont.load_default()


def fit_font_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path | None,
    box_width: int,
    box_height: int,
) -> tuple[ImageFont.ImageFont, int, tuple[int, int, int, int]]:
    if box_width <= 1 or box_height <= 1 or not text.strip():
        fallback_font = ImageFont.load_default()
        fallback_bbox = draw.textbbox((0, 0), text, font=fallback_font)
        return fallback_font, 10, fallback_bbox

    lo, hi = 6, max(6, int(box_height * 2.0))
    best_font = load_font_for_size(font_path, lo)
    best_size = lo
    best_bbox = draw.textbbox((0, 0), text, font=best_font)

    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font_for_size(font_path, mid)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= box_width and text_h <= box_height:
            best_font = font
            best_size = mid
            best_bbox = bbox
            lo = mid + 1
        else:
            hi = mid - 1

    return best_font, best_size, best_bbox


def render_text_back_to_image(
    image: Image.Image,
    ocr_items: list[dict[str, Any]],
    font_files: list[Path],
) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)

    for item in ocr_items:
        text_to_render = str(item.get("translated_text") or item.get("text") or "").strip()
        polygon = item.get("polygon", [])
        if not text_to_render or not isinstance(polygon, list):
            continue

        bbox = polygon_to_bbox(polygon, rendered.width, rendered.height)
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        box_w = right - left
        box_h = bottom - top

        font_path = resolve_render_font_path(item, font_files)
        color = resolve_render_color(item)
        font, font_size, text_bbox = fit_font_to_box(
            draw=draw,
            text=text_to_render,
            font_path=font_path,
            box_width=box_w,
            box_height=box_h,
        )

        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        x = left + max(0, (box_w - text_w) // 2) - text_bbox[0]
        y = top + max(0, (box_h - text_h) // 2) - text_bbox[1]
        draw.text((x, y), text_to_render, font=font, fill=color)

        item["render"] = {
            "text": text_to_render,
            "box": [left, top, right, bottom],
            "font_size": int(font_size),
            "font_file_used": None if font_path is None else font_path.name,
            "color_rgb_used": [int(color[0]), int(color[1]), int(color[2])],
        }

    return rendered


def main() -> None:
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    ds = load_dataset(args.dataset, split=args.split)

    total = len(ds)
    sample_n = min(args.num_samples, total)
    indices = list(range(total))
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(indices)
    indices = indices[:sample_n]

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    font_dir: Path = args.font_dir
    font_dir.mkdir(parents=True, exist_ok=True)
    font_files = discover_font_files(font_dir, args.font_match_max_fonts)
    if not font_files:
        print(
            f"[warn] No font files found in {font_dir}. "
            "Font matching will be skipped until you add .ttf/.otf/.ttc files."
        )
    else:
        print(f"[info] Loaded {len(font_files)} candidate fonts from: {font_dir}")

    ocr = build_ocr_model(args)
    translator = build_translator(args.source_lang, args.target_lang)
    inpaint_pipe = build_inpaint_pipeline(args)

    summary: list[dict[str, Any]] = []

    try:
        for out_idx, ds_idx in enumerate(indices):
            row = ds[ds_idx]

            image = row["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")

            sample_dir = output_dir / f"sample_{out_idx:04d}_idx_{ds_idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            input_image_path = sample_dir / "input.png"
            image.save(input_image_path)

            results = ocr.predict(
                str(input_image_path),
                use_layout_detection=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                prompt_label="spotting",
                max_new_tokens=args.max_new_tokens,
            )

            if not results:
                raise RuntimeError(f"No OCR result returned for dataset index {ds_idx}")
            result = results[0]

            spotting_res = result.get("spotting_res", {}) if hasattr(result, "get") else {}
            rec_polys = spotting_res.get("rec_polys", []) if isinstance(spotting_res, dict) else []
            rec_texts_raw = spotting_res.get("rec_texts", []) if isinstance(spotting_res, dict) else []
            rec_texts = [safe_text(t) for t in rec_texts_raw]

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            gt_bbox = denormalize_bbox_if_needed(row.get("bbox"), annotated.width, annotated.height)
            if gt_bbox:
                draw.rectangle(gt_bbox, outline=(0, 255, 0), width=3)

            ocr_items: list[dict[str, Any]] = []
            for i, (poly, text) in enumerate(zip(rec_polys, rec_texts)):
                points = normalize_polygon(poly)
                if not points:
                    continue

                draw.line(points + [points[0]], fill=(255, 0, 0), width=2)
                first_x, first_y = points[0]
                draw.text((first_x + 2, first_y + 2), str(i), fill=(255, 0, 0))

                ocr_items.append(
                    {
                        "id": i,
                        "text": text,
                        "polygon": [[float(x), float(y)] for x, y in points],
                    }
                )

            for item in ocr_items:
                style = extract_text_style(
                    image=image,
                    text=item["text"],
                    polygon=item["polygon"],
                    font_files=font_files,
                )
                item["text_color"] = style["text_color"]
                item["font_match"] = style["font_match"]

            annotated_path = sample_dir / "annotated.png"
            annotated.save(annotated_path)
            before_inpaint_path = sample_dir / "before_inpaint.png"
            image.save(before_inpaint_path)

            extracted_text_path = sample_dir / "extracted_text.txt"
            extracted_text_path.write_text(
                "\n".join(item["text"] for item in ocr_items),
                encoding="utf-8",
            )
            translated_texts = translate_texts(
                [item["text"] for item in ocr_items],
                translator,
            )
            if args.target_lang:
                for item, translated_text in zip(ocr_items, translated_texts):
                    item["translated_text"] = translated_text
                (sample_dir / "translated_text.txt").write_text(
                    "\n".join(translated_texts),
                    encoding="utf-8",
                )

            inpaint_mask = build_text_mask(
                image_size=image.size,
                ocr_items=ocr_items,
                mask_padding=args.mask_padding,
            )
            inpaint_mask.save(sample_dir / "inpaint_mask.png")
            after_inpaint = inpaint_text_regions(
                image=image,
                mask=inpaint_mask,
                inpaint_pipe=inpaint_pipe,
                args=args,
            )
            after_inpaint_path = sample_dir / "after_inpaint.png"
            after_inpaint.save(after_inpaint_path)
            final_rendered = render_text_back_to_image(
                image=after_inpaint,
                ocr_items=ocr_items,
                font_files=font_files,
            )
            final_rendered_path = sample_dir / "final_rendered.png"
            final_rendered.save(final_rendered_path)

            metadata = {
                "dataset": args.dataset,
                "split": args.split,
                "dataset_index": ds_idx,
                "file_name": row.get("file_name"),
                "instruction": row.get("instruction"),
                "data_type": row.get("data_type"),
                "data_source": row.get("data_source"),
                "ground_truth_bbox": row.get("bbox"),
                "ocr_item_count": len(ocr_items),
                "ocr_items": ocr_items,
                "source_lang": args.source_lang,
                "resolved_source_lang": None if translator is None else translator["source_lang"],
                "target_lang": args.target_lang,
                "font_dir": str(font_dir),
                "font_candidate_count": len(font_files),
                "inpainting_enabled": not args.disable_inpainting,
                "inpaint_backend": None if args.disable_inpainting else args.inpaint_backend,
                "inpaint_model_id": (
                    None if args.disable_inpainting or args.inpaint_backend != "sd" else args.inpaint_model_id
                ),
                "inpaint_tile_size": (
                    None if args.disable_inpainting or args.inpaint_backend != "sd" else args.inpaint_tile_size
                ),
                "before_inpaint_image": str(before_inpaint_path),
                "after_inpaint_image": str(after_inpaint_path),
                "final_rendered_image": str(final_rendered_path),
            }
            (sample_dir / "ocr.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            summary.append(metadata)

            print(
                f"[{out_idx + 1}/{sample_n}] idx={ds_idx} file={row.get('file_name')} "
                f"ocr_boxes={len(ocr_items)} -> {sample_dir}"
            )
    finally:
        ocr.close()

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Exported {len(summary)} samples to: {output_dir}")


if __name__ == "__main__":
    main()
