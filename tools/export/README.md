# ONNX Export Pipeline

Reproducible export of the on-device inference models to ONNX for **ONNX Runtime
Mobile (Android)**. This folder only *produces and validates* ONNX models — it
does **not** integrate anything into the Android app.

Two models are handled:

| Model | Source repo | Export mechanism | ONNX artifact(s) |
|-------|-------------|------------------|------------------|
| **YOLOe** `yoloe-11s-seg-pf.pt` | [Ultralytics](https://docs.ultralytics.com/modes/export/) | **Official** exporter `YOLO.export(format="onnx")` | `yoloe-11s-seg-pf.onnx` |
| **EfficientSAM** `efficient_sam_vits.pt` | [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM) (Meta) | **Custom** `torch.onnx.export()` (no official one-command exporter) | `efficient_sam_vits_encoder.onnx` + `efficient_sam_vits_decoder.onnx` |

### Why YOLOe uses the official exporter and EfficientSAM does not
- **YOLOe** ships inside the `ultralytics` package, whose `export()` already emits
  a correct, ORT-friendly ONNX graph (seg head + prototype masks). Re-implementing
  it by hand would be strictly worse, so we call the official path.
- **EfficientSAM**'s installed `efficient_sam` package exposes only `nn.Module`
  builders — **no** exporter. Upstream provides an ONNX *inference* example
  ([`EfficientSAM_onnx_example.py`](https://github.com/yformer/EfficientSAM/blob/main/EfficientSAM_onnx_example.py))
  that consumes a **split encoder + decoder** pair, so we follow that same split
  with a custom `torch.onnx.export()`.

> **Pre-exported ONNX already exists for EfficientSAM.** If you don't need to
> re-export, community ONNX builds are published by Kentaro Wada
> ([wkentaro/efficient-sam](https://github.com/wkentaro/efficient-sam), used by
> Labelme) and there are ready-to-run samples at
> [Kazuhito00/EfficientSAM-ONNX-Sample](https://github.com/Kazuhito00/EfficientSAM-ONNX-Sample).
> No pre-exported ONNX is published for the **YOLOe** `-pf` checkpoint — you must
> export it yourself (done here).

---

## 1. Dependencies

Already added to the project `.venv`. To reproduce from scratch:

```bash
# core (already in pyproject.toml)
uv pip install "torch>=2.13" torchvision ultralytics opencv-python numpy

# export + validation toolchain
uv pip install onnx onnxruntime onnxslim onnxscript
```

| Package | Used for |
|---------|----------|
| `ultralytics` | Official YOLOe → ONNX export |
| `torch` / `torchvision` | Loading checkpoints, `torch.onnx.export` (EfficientSAM) |
| `efficient_sam` | EfficientSAM model definition (installed from the Meta repo) |
| `onnx` | `onnx.checker` validation, graph inspection |
| `onnxslim` | Graph simplification (Ultralytics `simplify=True`) |
| `onnxscript` | Required by the torch 2.9+ ONNX exporter import path |
| `onnxruntime` | Verification inference + ORT-Mobile conversion check |
| `opencv-python` | Image I/O / resize in the verification script |

---

## 2. Export commands

Run from the **repo root** (`Yolo-test/`).

```bash
# YOLOe (official Ultralytics exporter)
python tools/export/export_yoloe_onnx.py \
    --weights models/objects365/yoloe-11s-seg-pf.pt \
    --imgsz 640 --opset 17

# EfficientSAM (custom encoder + decoder export)
python tools/export/export_efficientsam_onnx.py \
    --checkpoint efficient_sam_vits.pt --opset 17
```

Each script loads the model, exports, runs `onnx.checker`, and prints the input /
output **names and shapes**. Outputs are written to `models/onnx/`.

---

## 3. Validation / verification commands

```bash
# End-to-end smoke test in ONNX Runtime on a real image
python tools/export/verify_onnx.py --image images/Desk.jpeg --model all

# (Optional) Definitive ONNX Runtime *Mobile* compatibility check:
# converts to the .ort format the mobile runtime consumes. Success => all ops
# are supported by ORT Mobile. Output is throwaway; not committed.
python -m onnxruntime.tools.convert_onnx_models_to_ort models/onnx \
    --optimization_style Fixed --output_dir /tmp/ort_check
```

`validate_onnx()` (in `_common.py`) runs `onnx.checker`; `verify_onnx.py` runs a
real forward pass and prints the concrete output shapes.

---

## 4. Expected output files (`models/onnx/`)

| File | Size (approx) | Notes |
|------|---------------|-------|
| `yoloe-11s-seg-pf.onnx` | ~46 MB | single self-contained file |
| `efficient_sam_vits_encoder.onnx` | ~85 MB | single self-contained file |
| `efficient_sam_vits_decoder.onnx` | ~16 MB | single self-contained file |

All are single-file (weights embedded, **no** `.onnx.data` side-car).

### Tensor contracts

**YOLOe** (`imgsz=640`, static):
- input `images` : `float32 [1, 3, 640, 640]`, RGB, values `0..1`
- output `output0` : `float32 [1, 4621, 8400]` — detection/seg head
  (per-anchor: 4 box + class scores + 32 mask coefficients)
- output `output1` : `float32 [1, 32, 160, 160]` — mask prototypes
  (combine with the 32 coefficients to build instance masks)

**EfficientSAM encoder**:
- input `batched_images` : `float32 [1, 3, 1024, 1024]`, RGB, values `0..1`
  (ImageNet mean/std normalization is **inside** the graph)
- output `image_embeddings` : `float32 [1, 256, 64, 64]`

**EfficientSAM decoder**:
- input `image_embeddings` : `float32 [1, 256, 64, 64]`
- input `batched_point_coords` : `float32 [1, Q, P, 2]`
- input `batched_point_labels` : `float32 [1, Q, P]`
- output `masks` : `float32 [1, Q, 3, 256, 256]` — low-res logits (>0 = foreground)
- output `iou_predictions` : `float32 [1, Q, 3]`

`Q` = number of prompt queries, `P` = points per query (both dynamic axes; batch
is dynamic too).

### Point-coordinate convention (decoder)
Coordinates are in the encoder's **1024×1024** space, so the host must scale from
original-image pixels:

```
x_1024 = x_orig * 1024 / orig_w
y_1024 = y_orig * 1024 / orig_h
```

Point labels: `1`=positive, `0`=negative, `2`=box top-left corner,
`3`=box bottom-right corner, `-1`=padding. Masks come back at 256×256 — upsample
(bilinear) to the original image and threshold at `0` on-device.

---

## 5. Common export issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: onnxscript` | torch ≥2.9 exporter import path needs it | `uv pip install onnxscript` |
| `No Adapter To Version 17 for Pad` + opset silently becomes 18 | New **dynamo** exporter targets opset 18 and can't down-convert `Pad` to 17 | EfficientSAM export uses `dynamo=False` (legacy TorchScript exporter) which honors opset 17 and writes a single file |
| Two files `model.onnx` + `model.onnx.data` | dynamo exporter externalizes weights | We use `dynamo=False`; if you must use dynamo, re-save with `onnx.save(..., save_as_external_data=False)` |
| `simplify=True` fails / no-ops | `onnxslim` missing | `uv pip install onnxslim` |
| Ultralytics writes ONNX next to the `.pt` | default export location | `export_yoloe_onnx.py` copies it into `models/onnx/` and removes ambiguity |
| `TracerWarning: Converting a tensor to a Python float/bool` (EfficientSAM) | expected — shape math folded to constants for the fixed 1024 input | harmless; encoder input is fixed at 1024×1024 by design |
| Wrong EfficientSAM weights key | checkpoint stores state under `["model"]` | handled automatically in `load_efficientsam()` |

---

## 6. Troubleshooting

- **Verify with ONNX Runtime, not just `onnx.checker`.** `checker` only validates
  the graph structure; `verify_onnx.py` proves it actually runs and yields sane
  shapes. Run it before trusting an export.
- **Encoder input size is fixed to 1024×1024** (only the batch axis is dynamic).
  Resize/letterbox on the host. If you need arbitrary input sizes, re-export with
  `H`/`W` as dynamic axes (slower graph).
- **Empty / all-foreground EfficientSAM masks** almost always mean the point
  coordinates were not scaled into 1024 space, or the box labels are wrong
  (`2`/`3` for the two corners).
- **ORT Mobile op-support failures**: run the `convert_onnx_models_to_ort` command
  in §3. It prints exactly which operator is unsupported. All three models here
  currently convert cleanly (3/3) at opset 17.
- **Re-exporting after a torch/ultralytics upgrade** can change the opset or graph
  layout — re-run `verify_onnx.py` and the `.ort` conversion after any upgrade.

---

## Files in this folder

- `export_yoloe_onnx.py` — YOLOe → ONNX (official exporter)
- `export_efficientsam_onnx.py` — EfficientSAM → ONNX (custom encoder + decoder)
- `verify_onnx.py` — ONNX Runtime smoke test on a sample image
- `_common.py` — shared validation / inspection helpers
