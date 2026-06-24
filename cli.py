"""
cli.py — Standardised entry point for the cat-detection container.

Subcommands
-----------
  info     Print /app/STUDENT.json to stdout and exit 0.
  predict  Run the ONNX detector on every image in /data/input/,
           write /data/output/predictions.csv, exit 0.

Usage (inside the container):
  python /app/app/cli.py info
  python /app/app/cli.py predict
"""

import json
import os
import sys
import csv
from pathlib import Path

# ── Paths baked into the container ─────────────────────────────────────────
STUDENT_JSON   = Path("/app/STUDENT.json")
ONNX_MODEL     = Path("/app/models/best.onnx")
INPUT_DIR      = Path("/data/input")
OUTPUT_DIR     = Path("/data/output")
OUTPUT_CSV     = OUTPUT_DIR / "predictions.csv"

IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CSV_HEADER     = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]


# ── Sub-command: info ───────────────────────────────────────────────────────

def cmd_info() -> int:
    """Print STUDENT.json to stdout as valid JSON, return exit code."""
    if not STUDENT_JSON.exists():
        print(f"ERROR: {STUDENT_JSON} not found inside container.", file=sys.stderr)
        return 1
    with open(STUDENT_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(json.dumps(data, indent=2))
    return 0


# ── Sub-command: predict ────────────────────────────────────────────────────

def cmd_predict() -> int:
    """
    Detect cats in every image under /data/input/, write predictions.csv.
    Returns 0 on success, 1 on fatal error.
    """
    if not ONNX_MODEL.exists():
        print(f"ERROR: ONNX model not found at {ONNX_MODEL}", file=sys.stderr)
        return 1

    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory {INPUT_DIR} does not exist.", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Lazy import so 'info' sub-command stays lightweight ────────────────
    from app.detector import CatDetector

    print(f"[cli] Loading model from {ONNX_MODEL} …", flush=True)
    detector = CatDetector(
        onnx_path    = str(ONNX_MODEL),
        imgsz        = 640,
        conf         = 0.25,
        class_names  = ("cat",),
    )

    # ── Collect all image paths (recursive) ────────────────────────────────
    image_paths = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        print(f"[cli] WARNING: no images found in {INPUT_DIR}", file=sys.stderr)

    print(f"[cli] Running inference on {len(image_paths)} image(s) …", flush=True)

    # ── Write CSV ───────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_HEADER)
        writer.writeheader()

        for img_path in image_paths:
            # image_path in CSV = relative path from /data/input/, forward slashes
            rel_path = img_path.relative_to(INPUT_DIR).as_posix()

            try:
                detections = detector.predict(str(img_path))
            except Exception as exc:
                print(f"[cli] WARNING: failed to process {rel_path}: {exc}",
                      file=sys.stderr)
                detections = []

            if not detections:
                # No detections → write one row with empty box fields
                writer.writerow({
                    "image_path": rel_path,
                    "xmin": "", "ymin": "", "xmax": "", "ymax": "",
                    "confidence": "", "class": "",
                })
            else:
                for det in detections:
                    writer.writerow({
                        "image_path": rel_path,
                        "xmin":       det["xmin"],
                        "ymin":       det["ymin"],
                        "xmax":       det["xmax"],
                        "ymax":       det["ymax"],
                        "confidence": det["confidence"],
                        "class":      det["class"],
                    })

    print(f"[cli] Written → {OUTPUT_CSV}", flush=True)
    return 0


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python /app/app/cli.py <info|predict>", file=sys.stderr)
        sys.exit(1)

    subcommand = sys.argv[1].lower()

    if subcommand == "info":
        sys.exit(cmd_info())
    elif subcommand == "predict":
        sys.exit(cmd_predict())
    else:
        print(f"Unknown subcommand '{subcommand}'. Use 'info' or 'predict'.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
