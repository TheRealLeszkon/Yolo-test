#!/usr/bin/env python
"""Verify the exported ONNX models with ONNX Runtime on a sample image.

For each model this:
  * loads the .onnx with onnxruntime (CPUExecutionProvider),
  * runs one inference on a real image,
  * confirms the graph executes,
  * prints the concrete output tensor shapes.

This is a smoke test (does it run + produce sane shapes), not an accuracy test.

Usage:
    python tools/export/verify_onnx.py --image images/Desk.jpeg --model all
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import ONNX_OUT_DIR  # noqa: E402

YOLOE = ONNX_OUT_DIR / "yoloe-11s-seg-pf.onnx"
ESAM_ENCODER = ONNX_OUT_DIR / "efficient_sam_vits_encoder.onnx"
ESAM_DECODER = ONNX_OUT_DIR / "efficient_sam_vits_decoder.onnx"


def load_rgb(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def session(path: pathlib.Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# YOLOe
# ---------------------------------------------------------------------------
def verify_yoloe(image_rgb: np.ndarray, imgsz: int = 640) -> bool:
    print(f"\n=== YOLOe : {YOLOE.name} ===")
    if not YOLOE.exists():
        print("  SKIP: model not found (run export_yoloe_onnx.py first)")
        return False
    sess = session(YOLOE)

    resized = cv2.resize(image_rgb, (imgsz, imgsz))
    blob = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # [1,3,H,W]

    name = sess.get_inputs()[0].name
    outs = sess.run(None, {name: blob})
    print(f"  input  '{name}': {blob.shape} {blob.dtype}")
    for o, arr in zip(sess.get_outputs(), outs):
        print(f"  output '{o.name}': {tuple(arr.shape)} {arr.dtype}")
    print("  YOLOe inference: OK")
    return True


# ---------------------------------------------------------------------------
# EfficientSAM (encoder + decoder)
# ---------------------------------------------------------------------------
def verify_efficientsam(image_rgb: np.ndarray, enc_size: int = 1024) -> bool:
    print(f"\n=== EfficientSAM : {ESAM_ENCODER.name} + {ESAM_DECODER.name} ===")
    if not (ESAM_ENCODER.exists() and ESAM_DECODER.exists()):
        print("  SKIP: encoder/decoder not found (run export_efficientsam_onnx.py first)")
        return False

    orig_h, orig_w = image_rgb.shape[:2]
    enc = session(ESAM_ENCODER)
    dec = session(ESAM_DECODER)

    # --- encoder: raw RGB in 0..1 at 1024x1024 (normalization is inside the graph) ---
    img = cv2.resize(image_rgb, (enc_size, enc_size)).astype(np.float32)
    img = img.transpose(2, 0, 1)[None] / 255.0  # [1,3,1024,1024]
    embeddings = enc.run(None, {"batched_images": img})[0]
    print(f"  encoder input  'batched_images': {img.shape} {img.dtype}")
    print(f"  encoder output 'image_embeddings': {tuple(embeddings.shape)} {embeddings.dtype}")

    # --- decoder: box prompt (two corners), coordinates in 1024 encoder space ---
    # Use a central box of the original image, scaled into 1024 space.
    def to_1024(x, y):
        return [x * enc_size / orig_w, y * enc_size / orig_h]

    x1, y1 = orig_w * 0.25, orig_h * 0.25
    x2, y2 = orig_w * 0.75, orig_h * 0.75
    coords = np.array([[[to_1024(x1, y1), to_1024(x2, y2)]]], dtype=np.float32)  # [1,1,2,2]
    labels = np.array([[[2.0, 3.0]]], dtype=np.float32)  # 2=top-left, 3=bottom-right

    masks, iou = dec.run(
        None,
        {
            "image_embeddings": embeddings,
            "batched_point_coords": coords,
            "batched_point_labels": labels,
        },
    )
    print(f"  decoder input  'batched_point_coords': {coords.shape} {coords.dtype}")
    print(f"  decoder output 'masks': {tuple(masks.shape)} {masks.dtype}")
    print(f"  decoder output 'iou_predictions': {tuple(iou.shape)} {iou.dtype}")

    # Pick best mask and upsample to original size to prove it is usable.
    best = int(np.argmax(iou[0, 0]))
    low_res = masks[0, 0, best]  # [256,256] logits
    full = cv2.resize(low_res, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary = full > 0
    print(f"  best mask idx={best} iou={iou[0,0,best]:.3f} "
          f"positive_px={int(binary.sum())}/{binary.size}")
    print("  EfficientSAM inference: OK")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="images/Desk.jpeg", help="Sample image path.")
    ap.add_argument("--model", choices=["yoloe", "efficientsam", "all"], default="all")
    args = ap.parse_args()

    image_rgb = load_rgb(args.image)
    print(f"Sample image: {args.image}  ({image_rgb.shape[1]}x{image_rgb.shape[0]})")

    ok = True
    if args.model in ("yoloe", "all"):
        ok &= verify_yoloe(image_rgb)
    if args.model in ("efficientsam", "all"):
        ok &= verify_efficientsam(image_rgb)

    print("\n" + ("All requested models verified: OK" if ok
                  else "Some models were skipped or failed (see above)."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
