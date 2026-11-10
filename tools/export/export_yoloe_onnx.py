#!/usr/bin/env python
"""Export the YOLOe segmentation model (yoloe-11s-seg-pf.pt) to ONNX.

Mechanism: OFFICIAL Ultralytics exporter (`YOLO.export(format="onnx")`).
YOLOe ships inside the `ultralytics` package, so we do NOT hand-roll a
torch.onnx.export() call -- the official exporter already handles the
seg-head post-processing, NMS-free layout and opset selection correctly.

Docs: https://docs.ultralytics.com/modes/export/

The exporter writes `<model>.onnx` next to the .pt file; this script then
copies it into models/onnx/, validates it, and prints its I/O signature.

Usage:
    python tools/export/export_yoloe_onnx.py \
        --weights models/objects365/yoloe-11s-seg-pf.pt \
        --imgsz 640 --opset 17
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import describe_onnx_io, onnx_out_dir, onnx_size_mb, validate_onnx  # noqa: E402

DEFAULT_WEIGHTS = "models/objects365/yoloe-11s-seg-pf.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="Path to the YOLOe .pt checkpoint.")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Square export resolution (default 640).")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset (default 17, ORT-Mobile friendly).")
    ap.add_argument("--simplify", action="store_true", default=True,
                    help="Run onnxslim graph simplification (default on).")
    ap.add_argument("--no-simplify", dest="simplify", action="store_false")
    ap.add_argument("--dynamic", action="store_true", default=False,
                    help="Export dynamic axes (default off -> static, best for mobile).")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = pathlib.Path(args.weights).resolve()
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 1

    print(f"Loading YOLOe model: {weights}")
    model = YOLO(str(weights))
    print(f"  task = {model.task}")

    print(f"Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset}, "
          f"simplify={args.simplify}, dynamic={args.dynamic}) ...")
    exported = model.export(
        format="onnx",
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        imgsz=args.imgsz,
    )
    exported = pathlib.Path(exported).resolve()

    dest = onnx_out_dir() / f"{weights.stem}.onnx"
    if exported != dest:
        shutil.copy2(exported, dest)
    print(f"\nSaved: {dest}  ({onnx_size_mb(dest):.1f} MB)")

    print("\nValidating exported model ...")
    validate_onnx(dest)
    describe_onnx_io(dest)

    print("\nYOLOe ONNX export: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
