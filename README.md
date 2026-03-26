# Image Translation Benchmark (ScreenSpot + PaddleOCR-VL)

This project builds a complete UI text replacement pipeline on top of the
`rootsautomation/ScreenSpot` dataset.

It performs OCR, translation, text removal (inpainting), text style extraction
(color + font family), and final text re-rendering back into each detected
region.

Main entrypoint:
- `src/export_screenspot_paddleocr_vl.py`

## Example Result

english to indonesian language

<p>
  <img src="examples/sample_0004_before_inpaint.png" alt="Before Example" width="200" />
  <img src="examples/sample_0004_final_rendered.png" alt="Final Example" width="200" />
</p>

## What It Does

- Samples `N` images from ScreenSpot.
- Runs OCR spotting with `PaddleOCR-VL-1.5`.
- Extracts text style per OCR region.
- Estimates text color from thresholded text pixels.
- Matches font family by SSIM against candidate fonts.
- Removes original text using inpainting (`lama` or `sd` backend).
- Renders translated text back into each OCR box with auto-fitted font size.
- Exports all artifacts and metadata per sample.

## Project Layout

```text
.
├── src/
│   ├── export_screenspot_paddleocr_vl.py   # Main pipeline
│   ├── font_download.py                     # Optional helper for downloading fonts
│   └── model_test.py                        # Optional environment check
├── examples/                                # README example images
├── assets/
│   └── fonts/                               # Candidate fonts (recursive scan)
└── outputs/
    └── screenspot_vl/                       # Generated results
```

## Requirements

- Python `>=3.11`
- `pip`
- `requirements.txt` in project root
- This project runs on cuda 12.6, other cuda version can be checked manually

All Python dependencies are installed from `requirements.txt`.

## Installation

### Option A: `venv` (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda create -n image-translation python=3.11 -y
conda activate image-translation
pip install --upgrade pip
pip install -r requirements.txt
```

### Notes

- `requirements.txt` already includes `argostranslate`, `simple-lama-inpainting`, and deep learning dependencies.
- If your machine/GPU differs, you may need to adjust some pinned packages (especially `torch` and CUDA-related entries).
- Changes 'paddlepaddle-gpu @ file:///' in the `requirements.txt` to match the .whl file path.
## Fonts Setup

Place font files in `assets/fonts/` before running font matching/rendering.

Supported scan extensions:
- `.ttf`
- `.otf`
- `.ttc`
- `.woff`
- `.woff2`

The script scans `--font-dir` recursively, so nested family folders are fine.
For best compatibility with rendering, prefer `.ttf`/`.otf`/`.ttc`.

## Quick Start

```bash
python src/export_screenspot_paddleocr_vl.py \
  --num-samples 10 \
  --output-dir outputs/screenspot_vl
```

Current script defaults:
- `--target-lang id`
- `--inpaint-backend lama`
- `--font-dir assets/fonts`

## Common Commands

Run with LaMa backend (full-image inpainting):

```bash
python src/export_screenspot_paddleocr_vl.py \
  --num-samples 10 \
  --inpaint-backend lama
```

Run with Stable Diffusion backend:

```bash
python src/export_screenspot_paddleocr_vl.py \
  --num-samples 10 \
  --inpaint-backend sd \
  --inpaint-model-id stable-diffusion-v1-5/stable-diffusion-inpainting
```

Set target translation language:

```bash
python src/export_screenspot_paddleocr_vl.py \
  --num-samples 10 \
  --target-lang en
```

Use custom font directory and cap candidate fonts:

```bash
python src/export_screenspot_paddleocr_vl.py \
  --num-samples 10 \
  --font-dir assets/fonts \
  --font-match-max-fonts 100
```

## Main CLI Options

- `--num-samples / -n`: number of samples to process (required)
- `--dataset`: HF dataset id (default: `rootsautomation/ScreenSpot`)
- `--split`: dataset split (default: `test`)
- `--output-dir`: output directory
- `--seed`: random seed
- `--no-shuffle`: use first `N` rows instead of random sampling
- `--device`: Paddle device (`cpu`, `gpu:0`, ...)
- `--max-new-tokens`: OCR max tokens
- `--source-lang`: translation source (default: `auto`)
- `--target-lang`: translation destination (default: `id`)
- `--disable-inpainting`: skip inpainting
- `--inpaint-backend`: `lama` or `sd`
- `--inpaint-model-id`: SD model id
- `--inpaint-prompt`: SD prompt
- `--inpaint-negative-prompt`: SD negative prompt
- `--inpaint-steps`: SD steps
- `--inpaint-guidance-scale`: SD guidance
- `--inpaint-strength`: SD strength
- `--mask-padding`: inpaint mask expansion
- `--inpaint-tile-size`: tile size for SD backend
- `--font-dir`: candidate fonts directory (recursive scan)
- `--font-match-max-fonts`: max fonts to evaluate

Run full argument help:

```bash
python src/export_screenspot_paddleocr_vl.py --help
```

## Output Artifacts

For each sampled item (`sample_xxxx_idx_xxxx/`), generated files include:

- `input.png`
- `annotated.png`
- `extracted_text.txt`
- `translated_text.txt` (if translation enabled)
- `inpaint_mask.png`
- `before_inpaint.png`
- `after_inpaint.png`
- `final_rendered.png`
- `ocr.json`

A global `summary.json` is written in the output root directory.

## Metadata Highlights (`ocr.json`)

Per OCR item includes:
- `text`, `translated_text`
- `polygon`
- `text_color` (`median_rgb`, `mean_rgb`, hex variants)
- `font_match` (`font_family`, `font_file`, `font_path`, `ssim`)
- `render` (`text`, box, font size used, color used)

## Optional Utilities

Download starter fonts:

```bash
python src/font_download.py
```

Run a simple model environment check:

```bash
python src/model_test.py
```

## Troubleshooting

- Missing dependencies: re-run `pip install -r requirements.txt`.
- No fonts found warning: add files under `assets/fonts/` or set `--font-dir`.
- OCR or GPU issues: try `--device cpu` and verify Paddle/CUDA setup.
