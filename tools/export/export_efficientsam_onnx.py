#!/usr/bin/env python
"""Export EfficientSAM (efficient_sam_vits.pt) to ONNX.

Mechanism: CUSTOM torch.onnx.export().
The installed `efficient_sam` package (Meta / yformer) ships NO official
one-command ONNX exporter -- its __init__ only builds nn.Modules, and the
upstream repo provides an ONNX *inference* example that consumes a
pre-split encoder + decoder pair (EfficientSAM_onnx_example.py). We therefore
follow that same split and export two graphs:

  1. efficient_sam_vits_encoder.onnx
        in : batched_images            [1, 3, 1024, 1024]  float32  (RGB, 0..1)
        out: image_embeddings          [1, 256, 64, 64]    float32

  2. efficient_sam_vits_decoder.onnx
        in : image_embeddings          [1, 256, 64, 64]    float32
             batched_point_coords      [1, Q, P, 2]        float32  (encoder / 1024 space)
             batched_point_labels      [1, Q, P]           float32
        out: masks                     [1, Q, 3, 256, 256] float32  (low-res logits)
             iou_predictions           [1, Q, 3]           float32

Point-coordinate convention
---------------------------
The decoder consumes point coordinates already expressed in the encoder's
1024x1024 input space:  x_1024 = x_orig * 1024 / orig_w   (same for y).
This keeps the graph fully static (no runtime image-size tensor) which is the
friendliest form for ONNX Runtime Mobile. Masks come back at the native 256x256
decoder resolution; upsample to the original image on-device with bilinear +
threshold at 0.  Padding sentinel value for unused points is -1.

Usage:
    python tools/export/export_efficientsam_onnx.py \
        --checkpoint efficient_sam_vits.pt --opset 17
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import describe_onnx_io, onnx_out_dir, onnx_size_mb, validate_onnx  # noqa: E402

DEFAULT_CHECKPOINT = "efficient_sam_vits.pt"
IMG_SIZE = 1024
EMBED_DIM = 256
EMBED_HW = 64  # 1024 / patch_size(16)


class EncoderWrapper(nn.Module):
    """Image -> image embeddings (runs only the ViT encoder + neck)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batched_images: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_embeddings(batched_images)


class DecoderWrapper(nn.Module):
    """Embeddings + point prompts -> low-res mask logits + IoU scores.

    Points are expected in 1024x1024 encoder space, so input_h/input_w are
    fixed to 1024 (get_rescaled_pts becomes an identity scaling). output_h/w
    are -1 so predict_masks returns the native 256x256 logits.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image_embeddings, batched_point_coords, batched_point_labels):
        masks, iou = self.model.predict_masks(
            image_embeddings,
            batched_point_coords,
            batched_point_labels,
            multimask_output=True,
            input_h=IMG_SIZE,
            input_w=IMG_SIZE,
            output_h=-1,
            output_w=-1,
        )
        return masks, iou


def load_efficientsam(checkpoint: pathlib.Path) -> nn.Module:
    from efficient_sam.build_efficient_sam import build_efficient_sam

    # vit-small variant (matches efficient_sam_vits.pt)
    model = build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=None,
    )
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def export_encoder(model: nn.Module, opset: int) -> pathlib.Path:
    out = onnx_out_dir() / "efficient_sam_vits_encoder.onnx"
    wrapper = EncoderWrapper(model).eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    print("\n[encoder] exporting ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out),
            input_names=["batched_images"],
            output_names=["image_embeddings"],
            opset_version=opset,
            dynamic_axes={
                "batched_images": {0: "batch"},
                "image_embeddings": {0: "batch"},
            },
            do_constant_folding=True,
            # Legacy TorchScript exporter: honors opset_version exactly and
            # embeds weights in a single self-contained .onnx (no .onnx.data),
            # which is the friendliest form for ONNX Runtime Mobile.
            dynamo=False,
        )
    print(f"[encoder] saved: {out}  ({onnx_size_mb(out):.1f} MB)")
    return out


def export_decoder(model: nn.Module, opset: int) -> pathlib.Path:
    out = onnx_out_dir() / "efficient_sam_vits_decoder.onnx"
    wrapper = DecoderWrapper(model).eval()

    # Example prompt: 1 query, 2 points (a box's two corners, labels 2 & 3),
    # coordinates already in 1024 space.
    embeddings = torch.randn(1, EMBED_DIM, EMBED_HW, EMBED_HW)
    coords = torch.tensor([[[[200.0, 300.0], [800.0, 700.0]]]], dtype=torch.float)
    labels = torch.tensor([[[2.0, 3.0]]], dtype=torch.float)

    print("\n[decoder] exporting ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (embeddings, coords, labels),
            str(out),
            input_names=[
                "image_embeddings",
                "batched_point_coords",
                "batched_point_labels",
            ],
            output_names=["masks", "iou_predictions"],
            opset_version=opset,
            dynamic_axes={
                "image_embeddings": {0: "batch"},
                "batched_point_coords": {0: "batch", 1: "num_queries", 2: "num_points"},
                "batched_point_labels": {0: "batch", 1: "num_queries", 2: "num_points"},
                "masks": {0: "batch", 1: "num_queries"},
                "iou_predictions": {0: "batch", 1: "num_queries"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[decoder] saved: {out}  ({onnx_size_mb(out):.1f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="Path to efficient_sam_vits.pt")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset (default 17, ORT-Mobile friendly).")
    ap.add_argument("--part", choices=["encoder", "decoder", "both"], default="both")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    print(f"Loading EfficientSAM (vit-s) from: {ckpt}")
    model = load_efficientsam(ckpt)

    produced: list[pathlib.Path] = []
    if args.part in ("encoder", "both"):
        produced.append(export_encoder(model, args.opset))
    if args.part in ("decoder", "both"):
        produced.append(export_decoder(model, args.opset))

    print("\nValidating exported model(s) ...")
    for p in produced:
        validate_onnx(p)
        describe_onnx_io(p)

    print("\nEfficientSAM ONNX export: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
