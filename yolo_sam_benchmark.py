"""
yolo_sam_benchmark.py
=====================

Benchmark a YOLO-detection -> SAM-segmentation pipeline.

The idea is to use a YOLO detection to *guide* SAM so that SAM segments only the
detected object, instead of running SAM automatically over the whole image:

    image -> YOLO detection -> bounding box -> SAM (box prompt) -> single mask

Every YOLO model is paired with every SAM model:

    YOLO : Objects365, LVIS, YOLO-World
    SAM  : FastSAM, EfficientSAM

For every image in ``images/`` and every YOLO/SAM combination the script:

    1. Runs YOLO and saves an image of the detections.
    2. Prompts SAM with each YOLO box and saves the segmented object as a
       transparent-background PNG cropped tightly around the mask.
    3. Builds a side-by-side comparison image (original | detections |
       +FastSAM | +EfficientSAM) per YOLO model.
    4. Records timing / count metrics to ``benchmark_results.csv``.

If any model fails to download, load, or run, the error is reported and the
remaining combinations are still benchmarked -- the run never aborts.

    python yolo_sam_benchmark.py
"""

import csv
import os
import re
import subprocess
import sys
import time
import urllib.request
import warnings


# -----------------------------------------------------------------------------
# Make sure the third-party packages we need are installed.
# -----------------------------------------------------------------------------
def ensure_packages():
    """Install missing pip packages so the script runs with minimal setup."""
    for import_name, pip_name in [("ultralytics", "ultralytics"),
                                  ("cv2", "opencv-python"),
                                  ("numpy", "numpy")]:
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing missing package: {pip_name} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name],
                           check=True)


ensure_packages()
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
IMAGES_DIR = "images"
OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"
CSV_PATH = "benchmark_results.csv"

# Minimum confidence for a YOLO detection to be kept.
CONF_THRESHOLD = 0.25

ULTRALYTICS_ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

# YOLO models. loader picks the ultralytics class that can read the weights.
#   * YOLO-World -> yolov8s-worldv2 (open-vocabulary detector).
#   * Objects365 -> YOLOE prompt-free model (pretrained on Objects365 + GoldG,
#                   detects its built-in vocabulary with no text prompt).
#   * LVIS       -> a YOLO-World variant carrying the LVIS vocabulary.
YOLO_MODELS = [
    {"name": "Objects365", "key": "objects365", "filename": "yoloe-11s-seg-pf.pt",
     "url": f"{ULTRALYTICS_ASSETS}/yoloe-11s-seg-pf.pt", "loader": "yoloe"},

    {"name": "LVIS", "key": "lvis", "filename": "yolov8s-worldv2-lvis.pt",
     "url": "https://huggingface.co/mbar0075/YOLO-World-LVIS/resolve/main/yolov8s-worldv2-lvis.pt",
     "loader": "world"},

    {"name": "YOLO-World", "key": "yoloworld", "filename": "yolov8s-worldv2.pt",
     "url": f"{ULTRALYTICS_ASSETS}/yolov8s-worldv2.pt", "loader": "world"},
]

# SAM models. ``existing`` lists paths where the weights may already sit (this
# repo ships both at its root) so we reuse them instead of re-downloading.
SAM_MODELS = [
    {"name": "FastSAM", "key": "fastsam", "filename": "FastSAM-x.pt",
     "url": f"{ULTRALYTICS_ASSETS}/FastSAM-x.pt", "existing": ["FastSAM-x.pt"]},

    {"name": "EfficientSAM", "key": "efficientsam", "filename": "efficient_sam_vits.pt",
     "url": None, "existing": ["efficient_sam_vits.pt"]},
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def pick_device():
    """Return the best available device: CUDA GPU > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    """Block until queued GPU work finishes, so timings are accurate."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def find_images():
    """Return sorted paths of every jpg/jpeg/png image in IMAGES_DIR."""
    extensions = (".jpg", ".jpeg", ".png")
    return [os.path.join(IMAGES_DIR, f) for f in sorted(os.listdir(IMAGES_DIR))
            if f.lower().endswith(extensions)]


def download_weights(spec):
    """Return a local path to the model weights, downloading them if needed.

    Reuses an ``existing`` copy or a previous download; otherwise fetches the
    file into ``models/<key>/``. Raises with a clear message if it cannot.
    """
    folder = os.path.join(MODELS_DIR, spec["key"])
    target = os.path.join(folder, spec["filename"])
    for candidate in [target] + spec.get("existing", []):
        if os.path.exists(candidate):
            return candidate
    if not spec.get("url"):
        raise FileNotFoundError(
            f"weights '{spec['filename']}' not found and no download URL is known; "
            f"please place the file at {target}")
    os.makedirs(folder, exist_ok=True)
    print(f"  downloading weights -> {target}")
    urllib.request.urlretrieve(spec["url"], target)
    return target


def safe_name(text):
    """Turn an arbitrary class label into a safe filename fragment."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "object"


# -----------------------------------------------------------------------------
# YOLO detection
# -----------------------------------------------------------------------------
def load_yolo(spec, device):
    """Load a YOLO model with the right ultralytics class, on ``device``."""
    from ultralytics import YOLO, YOLOE, YOLOWorld
    weights = download_weights(spec)
    if spec["loader"] == "world":
        model = YOLOWorld(weights)
    elif spec["loader"] == "yoloe":
        model = YOLOE(weights)
    else:
        model = YOLO(weights)
    model.to(device)
    return model


def detect(model, image_rgb, device):
    """Run YOLO and return (detections, annotated_bgr, inference_seconds).

    ``detections`` is a list of {label, confidence, bbox=[x1,y1,x2,y2]}.
    """
    sync(device)
    start = time.perf_counter()
    result = model.predict(image_rgb, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
    sync(device)
    infer_s = time.perf_counter() - start

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
    annotated_bgr = result.plot()  # BGR image with boxes + labels + confidence
    return detections, annotated_bgr, infer_s


# -----------------------------------------------------------------------------
# SAM segmenters (box prompt -> single mask). Each holds its model and the
# current image, so the expensive per-image work happens once in prepare().
# -----------------------------------------------------------------------------
class FastSamSegmenter:
    def __init__(self, weights, device):
        from ultralytics import FastSAM
        self.model = FastSAM(weights)
        self.device = device
        self.image_rgb = None

    def prepare(self, image_rgb):
        self.image_rgb = image_rgb

    def segment(self, box):
        """Segment the object inside ``box``; return a 2D boolean mask."""
        result = self.model.predict(self.image_rgb, bboxes=[box], device=self.device,
                                    retina_masks=True, verbose=False)[0]
        h, w = self.image_rgb.shape[:2]
        if result.masks is None or len(result.masks.data) == 0:
            return np.zeros((h, w), dtype=bool)
        mask = result.masks.data[0].cpu().numpy()
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)


class EfficientSamSegmenter:
    def __init__(self, weights, device):
        from efficient_sam.build_efficient_sam import build_efficient_sam
        model = build_efficient_sam(encoder_patch_embed_dim=384,
                                    encoder_num_heads=6, checkpoint=None)
        model.load_state_dict(torch.load(weights, map_location="cpu")["model"])
        model.to(device).eval()
        self.model = model
        self.device = device
        self.embedding = None  # image embedding, computed once per image
        self.size = None

    def prepare(self, image_rgb):
        h, w = image_rgb.shape[:2]
        self.size = (h, w)
        image_t = torch.from_numpy(image_rgb).float().permute(2, 0, 1)[None].to(self.device) / 255.0
        with torch.no_grad():
            self.embedding = self.model.get_image_embeddings(image_t)

    def segment(self, box):
        """Prompt EfficientSAM with the box corners; return a 2D boolean mask."""
        h, w = self.size
        x1, y1, x2, y2 = box
        # Two points = the box corners, labelled 2 (top-left) and 3 (bottom-right).
        points = torch.tensor([[[[x1, y1], [x2, y2]]]], dtype=torch.float, device=self.device)
        labels = torch.tensor([[[2, 3]]], dtype=torch.float, device=self.device)
        with torch.no_grad():
            # multimask returns several candidates; keep the highest-IoU one, which
            # is tight to the object. (The single-mask option leaks across the frame.)
            logits, iou = self.model.predict_masks(self.embedding, points, labels,
                                                   multimask_output=True,
                                                   input_h=h, input_w=w, output_h=h, output_w=w)
        best = int(torch.argmax(iou[0, 0]))
        return (logits[0, 0, best] > 0).cpu().numpy().astype(bool)


def load_sam(spec, device):
    """Build the right SAM segmenter for ``spec`` on ``device``."""
    weights = download_weights(spec)
    if spec["key"] == "fastsam":
        return FastSamSegmenter(weights, device)
    return EfficientSamSegmenter(weights, device)


# -----------------------------------------------------------------------------
# Visual outputs
# -----------------------------------------------------------------------------
def save_cutout(path, image_rgb, mask):
    """Save the masked object cropped tightly, with a transparent background."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return False
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    crop = image_rgb[y1:y2, x1:x2]
    alpha = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
    rgba = np.dstack([crop, alpha])
    cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return True


def overlay_masks(image_rgb, masks):
    """Return a BGR image with each mask blended in a distinct colour."""
    out = image_rgb.astype(np.float32).copy()
    rng = np.random.default_rng(0)
    for mask in masks:
        colour = rng.integers(64, 256, size=3).astype(np.float32)
        out[mask] = 0.45 * out[mask] + 0.55 * colour
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def make_comparison(path, panels, height=360):
    """Stack labelled BGR panels side by side into one comparison image."""
    tiles = []
    for title, image_bgr in panels:
        scale = height / image_bgr.shape[0]
        tile = cv2.resize(image_bgr, (int(image_bgr.shape[1] * scale), height))
        banner = np.zeros((32, tile.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([banner, tile]))
    max_h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, max_h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    cv2.imwrite(path, np.hstack(tiles))


# -----------------------------------------------------------------------------
# Benchmark driver
# -----------------------------------------------------------------------------
def main():
    device = pick_device()
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    image_paths = find_images()
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}/ (looking for jpg/jpeg/png).")
        return
    print(f"Found {len(image_paths)} image(s) in {IMAGES_DIR}/\n")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # --- Load every model once, remembering load times and any failures. ---
    yolo_loaded = {}   # key -> {"name", "model", "load_s"}
    for spec in YOLO_MODELS:
        try:
            sync(device)
            start = time.perf_counter()
            model = load_yolo(spec, device)
            sync(device)
            yolo_loaded[spec["key"]] = {"name": spec["name"], "model": model,
                                        "load_s": round(time.perf_counter() - start, 3)}
            print(f"[load] YOLO {spec['name']}: {yolo_loaded[spec['key']]['load_s']}s")
        except Exception as error:
            yolo_loaded[spec["key"]] = {"name": spec["name"], "model": None,
                                        "load_s": "", "error": str(error).splitlines()[0]}
            print(f"[fail] YOLO {spec['name']}: {str(error).splitlines()[0]}")

    sam_loaded = {}    # key -> {"name", "seg", "load_s"}
    for spec in SAM_MODELS:
        try:
            sync(device)
            start = time.perf_counter()
            segmenter = load_sam(spec, device)
            sync(device)
            sam_loaded[spec["key"]] = {"name": spec["name"], "seg": segmenter,
                                       "load_s": round(time.perf_counter() - start, 3)}
            print(f"[load] SAM  {spec['name']}: {sam_loaded[spec['key']]['load_s']}s")
        except Exception as error:
            sam_loaded[spec["key"]] = {"name": spec["name"], "seg": None,
                                       "load_s": "", "error": str(error).splitlines()[0]}
            print(f"[fail] SAM  {spec['name']}: {str(error).splitlines()[0]}")

    rows = []  # one row per (image, yolo, sam) combination

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[warn] could not read {image_path}, skipping")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"\n=== {image_name} ({image_rgb.shape[1]}x{image_rgb.shape[0]}) ===")

        out_dir = os.path.join(OUTPUTS_DIR, image_name)
        det_dir = os.path.join(out_dir, "detections")
        seg_root = os.path.join(out_dir, "segmentations")
        os.makedirs(det_dir, exist_ok=True)

        # Give each working SAM the image once (encodes it where relevant).
        for sam in sam_loaded.values():
            if sam["seg"] is not None:
                try:
                    sam["seg"].prepare(image_rgb)
                except Exception as error:
                    print(f"  [warn] {sam['name']} prepare failed: {str(error).splitlines()[0]}")
                    sam["ready"] = False
                else:
                    sam["ready"] = True

        for yspec in YOLO_MODELS:
            yolo = yolo_loaded[yspec["key"]]
            if yolo["model"] is None:
                for sspec in SAM_MODELS:
                    rows.append(failed_row(image_name, yolo, sam_loaded[sspec["key"]],
                                           f"YOLO load failed: {yolo.get('error', '')}"))
                continue

            # 1) Detect and save the detection image.
            try:
                detections, annotated_bgr, yolo_infer_s = detect(yolo["model"], image_rgb, device)
            except Exception as error:
                for sspec in SAM_MODELS:
                    rows.append(failed_row(image_name, yolo, sam_loaded[sspec["key"]],
                                           f"detection failed: {str(error).splitlines()[0]}"))
                print(f"  [fail] {yolo['name']} detection: {str(error).splitlines()[0]}")
                continue
            cv2.imwrite(os.path.join(det_dir, f"{yspec['key']}.jpg"), annotated_bgr)
            print(f"  {yolo['name']}: {len(detections)} detections ({yolo_infer_s:.3f}s)")

            panels = [("Original", image_bgr), ("Detections", annotated_bgr)]

            # 2) Segment each detection with each SAM.
            for sspec in SAM_MODELS:
                sam = sam_loaded[sspec["key"]]
                if sam["seg"] is None or not sam.get("ready", False):
                    rows.append(failed_row(image_name, yolo, sam,
                                           sam.get("error", "SAM not ready")))
                    continue

                combo_dir = os.path.join(seg_root, f"{yspec['key']}_{sspec['key']}")
                os.makedirs(combo_dir, exist_ok=True)
                seg_times = []
                masks = []
                n_segmented = 0
                for i, det in enumerate(detections):
                    try:
                        sync(device)
                        start = time.perf_counter()
                        mask = sam["seg"].segment(det["bbox"])
                        sync(device)
                        seg_times.append(time.perf_counter() - start)
                    except Exception as error:
                        print(f"    [warn] {sam['name']} on object {i} "
                              f"({det['label']}): {str(error).splitlines()[0]}")
                        continue
                    if mask.sum() == 0:
                        continue
                    masks.append(mask)
                    filename = f"{i:02d}_{safe_name(det['label'])}.png"
                    if save_cutout(os.path.join(combo_dir, filename), image_rgb, mask):
                        n_segmented += 1

                panels.append((f"+{sam['name']}", overlay_masks(image_rgb, masks)))

                sam_infer_s = sum(seg_times)
                avg_seg_ms = (sam_infer_s / len(seg_times) * 1000) if seg_times else 0
                rows.append({
                    "image": image_name,
                    "yolo": yolo["name"],
                    "sam": sam["name"],
                    "yolo_load_s": yolo["load_s"],
                    "sam_load_s": sam["load_s"],
                    "yolo_infer_s": round(yolo_infer_s, 3),
                    "sam_infer_s": round(sam_infer_s, 3),
                    "pipeline_total_s": round(yolo_infer_s + sam_infer_s, 3),
                    "detections": len(detections),
                    "segmented": n_segmented,
                    "avg_seg_ms": round(avg_seg_ms, 1),
                    "status": "ok",
                })
                print(f"    + {sam['name']}: segmented {n_segmented}/{len(detections)} "
                      f"objects, avg {avg_seg_ms:.0f}ms/obj")

            # 3) Comparison image for this image + YOLO model.
            make_comparison(os.path.join(out_dir, f"comparison_{yspec['key']}.jpg"), panels)

    print_table(rows)
    write_csv(rows)
    print(f"\nOutputs written to {OUTPUTS_DIR}/")
    print(f"Metrics written to {CSV_PATH}")


def failed_row(image_name, yolo, sam, status):
    """Build a metrics row for a combination that could not be run."""
    return {
        "image": image_name, "yolo": yolo["name"], "sam": sam["name"],
        "yolo_load_s": yolo.get("load_s", ""), "sam_load_s": sam.get("load_s", ""),
        "yolo_infer_s": "", "sam_infer_s": "", "pipeline_total_s": "",
        "detections": "", "segmented": "", "avg_seg_ms": "", "status": status,
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_table(rows):
    """Print a console summary aggregated per YOLO/SAM combination."""
    combos = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        key = (r["yolo"], r["sam"])
        agg = combos.setdefault(key, {"objects": 0, "seg_ms": [], "total": 0.0})
        agg["objects"] += r["segmented"]
        if r["avg_seg_ms"]:
            agg["seg_ms"].append(r["avg_seg_ms"])
        agg["total"] += r["pipeline_total_s"]

    headers = ["YOLO", "SAM", "Objects", "Avg Seg(ms)", "Total(s)"]
    table = [headers]
    for yspec in YOLO_MODELS:
        for sspec in SAM_MODELS:
            agg = combos.get((yspec["name"], sspec["name"]))
            if agg is None:
                table.append([yspec["name"], sspec["name"], "-", "-", "-"])
            else:
                avg_ms = sum(agg["seg_ms"]) / len(agg["seg_ms"]) if agg["seg_ms"] else 0
                table.append([yspec["name"], sspec["name"], str(agg["objects"]),
                              f"{avg_ms:.0f}", f"{agg['total']:.2f}"])

    widths = [max(len(row[c]) for row in table) for c in range(len(headers))]
    fmt = lambda cells: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    print("\n" + fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in table[1:]:
        print(fmt(row))
    print()


def write_csv(rows):
    """Write every per-(image, YOLO, SAM) row to a CSV file."""
    fields = ["image", "yolo", "sam", "yolo_load_s", "sam_load_s", "yolo_infer_s",
              "sam_infer_s", "pipeline_total_s", "detections", "segmented",
              "avg_seg_ms", "status"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
