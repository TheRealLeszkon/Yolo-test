"""
benchmark.py
============

Benchmark multiple Segment Anything (SAM) variants on the same input image.

For every model whose package is installed AND whose checkpoint file exists,
this script will:

    1. Load the model (and measure loading time).
    2. Run automatic mask generation on the input image (and measure
       inference time).
    3. Count the number of masks produced.
    4. Save an overlay image with every mask drawn in a random colour.
    5. Save every individual mask as its own PNG.
    6. Save a copy of the original image.

Results for all models are printed as a formatted table and written to a CSV.

Models that are not installed or whose checkpoints are missing are skipped
automatically -- the remaining models are still benchmarked.

Supported variants:
    * SAM ViT-B / ViT-L / ViT-H   (pip package: segment-anything)
    * MobileSAM                    (pip package: mobile-sam)
    * FastSAM                      (via ultralytics)
    * EfficientSAM                 (pip package: efficient-sam)
    * SAM2                         (pip package: sam2 / SAM-2)

Only the CONFIGURATION section below should need editing.
"""

# =============================================================================
#                              CONFIGURATION
# =============================================================================
# Edit ONLY the values in this section.

# Path to the image every model will be run on.
IMAGE_PATH = "images/Desk.jpeg"

# Directory where per-model output folders are written.
OUTPUT_DIR = "outputs"

# Directory where every model's overlay image is collected together, so all
# overlays for a run sit side by side for easy review.
OVERLAY_DIR = "overlays"

# Directory where a fresh, timestamped summary CSV is written per run.
CSV_DIR = "benchmarks"

# Checkpoint / config paths for each model.
# A model is skipped if the file(s) it needs do not exist on disk.
CHECKPOINTS = {
    "sam_vit_b":    "sam_vit_b_01ec64.pth",
    "sam_vit_l":    "sam_vit_l_0b3195.pth",
    "sam_vit_h":    "sam_vit_h_4b8939.pth",
    "mobilesam":    "mobile_sam.pt",
    "fastsam":      "FastSAM-x.pt",
    "efficientsam": "efficient_sam_vits.pt",
    # SAM2 needs a checkpoint AND a model-config (.yaml shipped with the sam2 pkg).
    "sam2":         "sam2.1_hiera_large.pt",
    "sam2_config":  "configs/sam2.1/sam2.1_hiera_l.yaml",
}

# =============================================================================
#                          END OF CONFIGURATION
# =============================================================================

import csv
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

import cv2 as cv
import numpy as np
import torch

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def get_device() -> str:
    """Pick the best available device: CUDA GPU > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device: str) -> None:
    """Block until queued GPU work on `device` has finished (for accurate timing)."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@contextmanager
def timer(device: str):
    """Context manager that measures wall-clock time.

    On a GPU it synchronises before and after so GPU work is fully counted.
    Usage:
        with timer(device) as get_elapsed:
            ...
        seconds = get_elapsed()
    """
    _sync(device)
    start = time.perf_counter()
    elapsed = {}
    yield lambda: elapsed["value"]
    _sync(device)
    elapsed["value"] = time.perf_counter() - start


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image from disk as an RGB numpy array."""
    image = cv.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def to_bool_masks(raw_masks) -> list:
    """Normalise any model's output into a list of 2D boolean numpy arrays.

    Accepts:
        * SAM-style list of dicts with a 'segmentation' key.
        * A list/array of 2D masks.
    """
    masks = []
    for m in raw_masks:
        if isinstance(m, dict):
            seg = m["segmentation"]
        else:
            seg = m
        masks.append(np.asarray(seg).astype(bool))
    return masks


# -----------------------------------------------------------------------------
# Output saving
# -----------------------------------------------------------------------------
def save_outputs(model_dir: str, image_rgb: np.ndarray, masks: list,
                 overlay_path: str = None) -> None:
    """Save the original image, a coloured overlay, and every mask PNG.

    If `overlay_path` is given, a copy of the overlay is also written there so
    every model's overlay can be collected together in one folder.
    """
    masks_dir = os.path.join(model_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # --- original image (RGB -> BGR for cv.imwrite) ---
    cv.imwrite(os.path.join(model_dir, "original.png"),
               cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR))

    # --- coloured overlay ---
    overlay = image_rgb.astype(np.float32).copy()
    rng = np.random.default_rng(0)  # fixed seed -> reproducible colours
    for seg in masks:
        colour = rng.integers(0, 256, size=3).astype(np.float32)
        overlay[seg] = 0.5 * overlay[seg] + 0.5 * colour  # 50 % blend
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_bgr = cv.cvtColor(overlay, cv.COLOR_RGB2BGR)
    cv.imwrite(os.path.join(model_dir, "overlay.png"), overlay_bgr)

    # --- collected copy of the overlay in the shared overlay folder ---
    if overlay_path is not None:
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        cv.imwrite(overlay_path, overlay_bgr)

    # --- individual masks (white where mask is True) ---
    for i, seg in enumerate(masks):
        mask_img = (seg.astype(np.uint8)) * 255
        cv.imwrite(os.path.join(masks_dir, f"mask_{i:03d}.png"), mask_img)


# -----------------------------------------------------------------------------
# Per-model loaders + mask generators
# -----------------------------------------------------------------------------
# Each function below returns a callable pair:
#     load()            -> a "generator" object (already on device)
#     generate(gen, im) -> a list of boolean masks
# They raise ImportError if the package is missing so the caller can skip them.

def build_sam(variant: str, checkpoint: str, device: str):
    """SAM ViT-B / ViT-L / ViT-H via the segment-anything package."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    def load():
        sam = sam_model_registry[variant](checkpoint=checkpoint)
        sam.to(device)
        return SamAutomaticMaskGenerator(sam)

    def generate(gen, image):
        return to_bool_masks(gen.generate(image))

    return load, generate


def build_mobilesam(checkpoint: str, device: str):
    """MobileSAM via the mobile_sam package (ViT-T backbone)."""
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    def load():
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        sam.to(device)
        sam.eval()
        return SamAutomaticMaskGenerator(sam)

    def generate(gen, image):
        return to_bool_masks(gen.generate(image))

    return load, generate


def build_fastsam(checkpoint: str, device: str):
    """FastSAM via the ultralytics package."""
    from ultralytics import FastSAM

    def load():
        return FastSAM(checkpoint)

    def generate(model, image):
        # Ultralytics wants BGR/np or a path; pass the RGB array, it handles it.
        results = model(image, device=device, retina_masks=True, verbose=False)
        result = results[0]
        if result.masks is None:
            return []
        # masks.data -> tensor (N, H, W); resize to the original resolution.
        h, w = image.shape[:2]
        out = []
        for m in result.masks.data.cpu().numpy():
            if m.shape != (h, w):
                m = cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
            out.append(m.astype(bool))
        return out

    return load, generate


def build_efficientsam(checkpoint: str, device: str):
    """EfficientSAM (ViT-S) with a simple point-grid automatic mask generator.

    EfficientSAM ships no built-in automatic mask generator, so we prompt it on
    a coarse grid of foreground points, keep the best confident mask per point,
    and drop near-duplicates by IoU. The image is encoded once and the whole
    grid is decoded in chunks, which keeps it reasonably fast.
    """
    from efficient_sam.build_efficient_sam import build_efficient_sam

    def load():
        # ViT-S architecture (same params as build_efficient_sam_vits), but built
        # without weights so we can load the checkpoint from the configured path.
        model = build_efficient_sam(
            encoder_patch_embed_dim=384,
            encoder_num_heads=6,
            checkpoint=None,
        )
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        model.to(device)
        model.eval()
        return model

    def generate(model, image, grid=16, chunk=16, min_iou=0.5, iou_dedup=0.85):
        h, w = image.shape[:2]
        img_t = torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255.0
        img_t = img_t.to(device)

        # Grid of foreground points in original-image pixel coordinates.
        ys = np.linspace(0, h - 1, grid, dtype=int)
        xs = np.linspace(0, w - 1, grid, dtype=int)
        pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)

        kept = []
        with torch.no_grad():
            emb = model.get_image_embeddings(img_t)  # encode the image once
            for i in range(0, len(pts), chunk):
                block = pts[i:i + chunk]                       # (n, 2)
                n = len(block)
                p = torch.from_numpy(block).to(device).view(1, n, 1, 2)
                lbl = torch.ones(1, n, 1, device=device)       # 1 = foreground
                logits, iou = model.predict_masks(
                    emb, p, lbl, multimask_output=True,
                    input_h=h, input_w=w, output_h=h, output_w=w,
                )
                # logits: (1, n, num_masks, h, w); iou: (1, n, num_masks)
                best = torch.argmax(iou[0], dim=1)             # (n,)
                for q in range(n):
                    b = int(best[q])
                    if float(iou[0, q, b]) < min_iou:
                        continue
                    seg = (logits[0, q, b] > 0).cpu().numpy().astype(bool)
                    if seg.sum() == 0:
                        continue
                    if any(_iou(seg, k) > iou_dedup for k in kept):
                        continue
                    kept.append(seg)
        return kept

    return load, generate


def build_sam2(checkpoint: str, config: str, device: str):
    """SAM2 via the sam2 package."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    def load():
        sam2 = build_sam2(config, checkpoint, device=device, apply_postprocessing=False)
        return SAM2AutomaticMaskGenerator(sam2)

    def generate(gen, image):
        return to_bool_masks(gen.generate(image))

    return load, generate


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


# -----------------------------------------------------------------------------
# Model registry: maps a folder name -> how to build it + what files it needs
# -----------------------------------------------------------------------------
def build_registry() -> dict:
    """Return an ordered mapping of model-key -> spec dict.

    Each spec has:
        'name'   : pretty display name
        'files'  : list of paths that must all exist for the model to run
        'factory': a callable (device) -> (load_fn, generate_fn)
    """
    C = CHECKPOINTS
    return {
        "sam_vit_b": {
            "name": "SAM ViT-B",
            "files": [C["sam_vit_b"]],
            "factory": lambda dev: build_sam("vit_b", C["sam_vit_b"], dev),
        },
        "sam_vit_l": {
            "name": "SAM ViT-L",
            "files": [C["sam_vit_l"]],
            "factory": lambda dev: build_sam("vit_l", C["sam_vit_l"], dev),
        },
        "sam_vit_h": {
            "name": "SAM ViT-H",
            "files": [C["sam_vit_h"]],
            "factory": lambda dev: build_sam("vit_h", C["sam_vit_h"], dev),
        },
        "mobilesam": {
            "name": "MobileSAM",
            "files": [C["mobilesam"]],
            "factory": lambda dev: build_mobilesam(C["mobilesam"], dev),
        },
        "fastsam": {
            "name": "FastSAM",
            "files": [C["fastsam"]],
            "factory": lambda dev: build_fastsam(C["fastsam"], dev),
        },
        "efficientsam": {
            "name": "EfficientSAM",
            "files": [C["efficientsam"]],
            "factory": lambda dev: build_efficientsam(C["efficientsam"], dev),
        },
        "sam2": {
            "name": "SAM2",
            "files": [C["sam2"]],  # config is bundled with the package
            "factory": lambda dev: build_sam2(C["sam2"], C["sam2_config"], dev),
        },
    }


# -----------------------------------------------------------------------------
# Benchmark driver
# -----------------------------------------------------------------------------
def _run_on_device(spec: dict, image_rgb: np.ndarray, device: str):
    """Build, load, and run one model on `device`.

    Returns (load_time, inference_time, masks). Raises on any failure so the
    caller can decide whether to fall back to another device.
    """
    load_fn, generate_fn = spec["factory"](device)

    with timer(device) as load_elapsed:
        generator = load_fn()
    load_time = round(load_elapsed(), 3)

    with timer(device) as infer_elapsed:
        masks = generate_fn(generator, image_rgb)
    infer_time = round(infer_elapsed(), 3)

    return load_time, infer_time, masks


def benchmark_model(key: str, spec: dict, image_rgb: np.ndarray, devices: list) -> dict:
    """Benchmark a single model, trying each device in `devices` until one works.

    Never raises: any failure is captured in the row's 'status' field so the
    remaining models can still be benchmarked. GPU backends (MPS in particular)
    have op-coverage gaps, so we transparently fall back to the next device.
    """
    row = {
        "model": spec["name"],
        "device": devices[0],
        "load_time_s": "",
        "inference_time_s": "",
        "num_masks": "",
        "status": "",
    }

    # 1) Skip early if any required file is missing.
    missing = [f for f in spec["files"] if not os.path.exists(f)]
    if missing:
        row["status"] = f"skipped (missing: {', '.join(missing)})"
        print(f"  [skip] {spec['name']}: {row['status']}")
        return row

    # 2) Try each candidate device in order.
    last_error = None
    for i, device in enumerate(devices):
        try:
            load_time, infer_time, masks = _run_on_device(spec, image_rgb, device)
        except ImportError as e:
            row["status"] = f"skipped (not installed: {e})"
            print(f"  [skip] {spec['name']}: {row['status']}")
            return row
        except Exception as e:  # noqa: BLE001 - keep the benchmark going
            last_error = e
            if i + 1 < len(devices):
                print(f"  [warn] {spec['name']}: failed on {device} "
                      f"({str(e).splitlines()[0]}); retrying on {devices[i + 1]}")
            continue

        # Success on this device.
        row["device"] = device
        row["load_time_s"] = load_time
        row["inference_time_s"] = infer_time
        row["num_masks"] = len(masks)

        # 3) Save all outputs.
        try:
            model_dir = os.path.join(OUTPUT_DIR, key)
            os.makedirs(model_dir, exist_ok=True)
            image_stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
            overlay_path = os.path.join(OVERLAY_DIR, f"{image_stem}_{key}.png")
            save_outputs(model_dir, image_rgb, masks, overlay_path=overlay_path)
        except Exception as e:  # noqa: BLE001
            row["status"] = f"ran but failed to save outputs ({e})"
            print(f"  [warn] {spec['name']}: {row['status']}")
            return row

        fallback = "" if device == devices[0] else f" (fell back to {device})"
        row["status"] = "ok" + fallback
        print(f"  [ ok ] {spec['name']}: {row['num_masks']} masks, "
              f"load {load_time}s, infer {infer_time}s on {device}{fallback}")
        return row

    # Every device failed.
    row["status"] = f"error ({last_error})"
    print(f"  [fail] {spec['name']}: {row['status']}")
    return row


def print_table(rows: list) -> None:
    """Pretty-print the benchmark results as an aligned table."""
    headers = ["Model", "Device", "Load (s)", "Infer (s)", "# Masks", "Status"]
    keys = ["model", "device", "load_time_s", "inference_time_s", "num_masks", "status"]

    table = [headers] + [[str(r[k]) for k in keys] for r in rows]
    widths = [max(len(r[c]) for r in table) for c in range(len(headers))]

    def fmt(cells):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    line = "-+-".join("-" * w for w in widths)
    print("\n" + fmt(headers))
    print(line)
    for r in table[1:]:
        print(fmt(r))
    print()


def write_csv(rows: list, path: str) -> None:
    """Write the benchmark results to a CSV file."""
    fields = ["model", "device", "load_time_s", "inference_time_s", "num_masks", "status"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    device = get_device()
    # Preferred device first; fall back to CPU if a model can't run on the GPU
    # (e.g. the SAM mask generators use float64, which Apple MPS doesn't support).
    devices = [device] if device == "cpu" else [device, "cpu"]
    print(f"Device: {device.upper()} (fallback: {' -> '.join(devices)})")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    print(f"Image:  {IMAGE_PATH}")
    image_rgb = load_image_rgb(IMAGE_PATH)
    print(f"Size:   {image_rgb.shape[1]}x{image_rgb.shape[0]}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    registry = build_registry()
    rows = []
    for key, spec in registry.items():
        rows.append(benchmark_model(key, spec, image_rgb, devices))

    print_table(rows)

    # Fresh, timestamped CSV per run/image so previous results are never
    # overwritten and can be reviewed later.
    os.makedirs(CSV_DIR, exist_ok=True)
    image_stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(CSV_DIR, f"benchmark_{image_stem}_{timestamp}.csv")
    write_csv(rows, csv_path)
    print(f"Overlays collected in {OVERLAY_DIR}/")
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
