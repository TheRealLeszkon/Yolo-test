"""
yolo_benchmark.py
=================

Benchmark several YOLO models on every image in the ``images/`` folder.

For each model this script:

    1. Downloads the weights into ``models/<name>/`` if they are not already there.
    2. Loads the model (measuring load time).
    3. Runs detection on every image (measuring inference time).
    4. Saves an annotated image (boxes + class labels + confidence) to
       ``outputs/<image_name>/<model>.jpg``.
    5. Writes the raw detections to ``outputs/<image_name>/<model>.json``.
    6. Records timing / detection / memory / model-size metrics.

At the end a summary table is printed and written to ``benchmark_results.csv``.

If a model cannot be downloaded, loaded, or run, the error is reported and the
remaining models are still benchmarked -- the whole run never aborts.

Run it with:

    python yolo_benchmark.py
"""

import csv
import json
import os
import subprocess
import sys
import time
import urllib.request


# -----------------------------------------------------------------------------
# Make sure the third-party packages we need are installed.
# -----------------------------------------------------------------------------
def ensure_packages():
    """Install missing pip packages so the script runs with minimal setup."""
    for import_name, pip_name in [("ultralytics", "ultralytics"),
                                  ("psutil", "psutil"),
                                  ("cv2", "opencv-python")]:
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing missing package: {pip_name} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name],
                           check=True)


ensure_packages()

import cv2
import psutil
import torch
from ultralytics import YOLO, YOLOE, YOLOWorld


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
IMAGES_DIR = "images"
OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"
CSV_PATH = "benchmark_results.csv"

# Minimum confidence for a detection to be kept.
CONF_THRESHOLD = 0.25

# Base URL for the official Ultralytics weight assets.
ULTRALYTICS_ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

# The models to benchmark. Each entry describes:
#   name    : pretty name used in the report and output filenames
#   subdir  : folder under models/ where the weights are stored
#   filename: weight filename
#   url     : where to download the weights from if they are missing
#   loader  : which ultralytics class loads the weights
#
# Notes on the less-obvious choices:
#   * YOLO-World  -> yolov8s-worldv2 (open-vocabulary detector).
#   * Objects365  -> YOLOE prompt-free model. YOLOE is pretrained on Objects365
#                    (+ GoldG) and detects its built-in vocabulary with no prompt.
#   * LVIS        -> a YOLO-World variant carrying the LVIS vocabulary.
MODELS = [
    {"name": "YOLO11n", "subdir": "yolo11n", "filename": "yolo11n.pt",
     "url": f"{ULTRALYTICS_ASSETS}/yolo11n.pt", "loader": "yolo"},

    {"name": "YOLO11s", "subdir": "yolo11s", "filename": "yolo11s.pt",
     "url": f"{ULTRALYTICS_ASSETS}/yolo11s.pt", "loader": "yolo"},

    {"name": "YOLO-World", "subdir": "yoloworld", "filename": "yolov8s-worldv2.pt",
     "url": f"{ULTRALYTICS_ASSETS}/yolov8s-worldv2.pt", "loader": "world"},

    {"name": "YOLO11-Seg", "subdir": "yolo11seg", "filename": "yolo11n-seg.pt",
     "url": f"{ULTRALYTICS_ASSETS}/yolo11n-seg.pt", "loader": "yolo"},

    {"name": "Objects365", "subdir": "objects365", "filename": "yoloe-11s-seg-pf.pt",
     "url": f"{ULTRALYTICS_ASSETS}/yoloe-11s-seg-pf.pt", "loader": "yoloe"},

    {"name": "LVIS", "subdir": "lvis", "filename": "yolov8s-worldv2-lvis.pt",
     "url": "https://huggingface.co/mbar0075/YOLO-World-LVIS/resolve/main/yolov8s-worldv2-lvis.pt",
     "loader": "world"},
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def pick_device():
    """Return the best available device: CUDA GPU > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_images():
    """Return sorted paths of every jpg/jpeg/png image in IMAGES_DIR."""
    extensions = (".jpg", ".jpeg", ".png")
    images = []
    for filename in sorted(os.listdir(IMAGES_DIR)):
        if filename.lower().endswith(extensions):
            images.append(os.path.join(IMAGES_DIR, filename))
    return images


def download_weights(model):
    """Return the local weights path, downloading it into models/ if needed."""
    folder = os.path.join(MODELS_DIR, model["subdir"])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, model["filename"])
    if os.path.exists(path):
        return path
    print(f"  downloading weights -> {path}")
    urllib.request.urlretrieve(model["url"], path)
    return path


def load_model(model, weights_path, device):
    """Load a model with the right ultralytics class and move it to the device."""
    if model["loader"] == "world":
        loaded = YOLOWorld(weights_path)
    elif model["loader"] == "yoloe":
        loaded = YOLOE(weights_path)
    else:
        loaded = YOLO(weights_path)
    loaded.to(device)
    return loaded


def read_vram_mb(device):
    """Return currently used GPU memory in MB, or None when unavailable."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device == "mps":
        try:
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
        except Exception:
            return None
    return None


def extract_detections(result):
    """Turn an ultralytics result into a list of {label, confidence, bbox} dicts."""
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": result.names[class_id],
            "confidence": round(confidence, 4),
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
        })
    return detections


# -----------------------------------------------------------------------------
# Benchmark one model across all images
# -----------------------------------------------------------------------------
def benchmark_model(model, image_paths, device):
    """Load and run one model on every image, returning a summary row dict.

    Never raises: any failure is captured in the row's 'status' field so the
    remaining models can still be benchmarked.
    """
    process = psutil.Process()
    row = {
        "model": model["name"],
        "load_s": "",
        "infer_s": "",
        "detections": "",
        "avg_conf": "",
        "categories": "",
        "ram_mb": "",
        "vram_mb": "",
        "size_mb": "",
        "status": "",
    }

    # 1) Download weights.
    try:
        weights_path = download_weights(model)
    except Exception as error:
        row["status"] = f"download failed ({error})"
        print(f"  [fail] {model['name']}: {row['status']}")
        return row
    row["size_mb"] = round(os.path.getsize(weights_path) / (1024 * 1024), 1)

    # 2) Load the model (on the preferred device, falling back to CPU).
    run_device = device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        start = time.perf_counter()
        loaded = load_model(model, weights_path, run_device)
        row["load_s"] = round(time.perf_counter() - start, 3)
    except Exception as error:
        if device != "cpu":
            print(f"  [warn] {model['name']}: load failed on {device} "
                  f"({str(error).splitlines()[0]}); retrying on CPU")
            try:
                run_device = "cpu"
                start = time.perf_counter()
                loaded = load_model(model, weights_path, run_device)
                row["load_s"] = round(time.perf_counter() - start, 3)
            except Exception as cpu_error:
                row["status"] = f"load failed ({cpu_error})"
                print(f"  [fail] {model['name']}: {row['status']}")
                return row
        else:
            row["status"] = f"load failed ({error})"
            print(f"  [fail] {model['name']}: {row['status']}")
            return row

    # 3) Run inference on every image and collect metrics.
    total_infer = 0.0
    all_confidences = []
    categories = set()
    peak_ram = process.memory_info().rss
    peak_vram = read_vram_mb(run_device)
    image_errors = 0

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(OUTPUTS_DIR, image_name)
        os.makedirs(out_dir, exist_ok=True)
        try:
            start = time.perf_counter()
            results = loaded.predict(image_path, conf=CONF_THRESHOLD,
                                     device=run_device, verbose=False)
            total_infer += time.perf_counter() - start
            result = results[0]

            detections = extract_detections(result)
            for detection in detections:
                all_confidences.append(detection["confidence"])
                categories.add(detection["label"])

            # Annotated image with boxes, labels and confidence values.
            annotated = result.plot()
            cv2.imwrite(os.path.join(out_dir, f"{model['subdir']}.jpg"), annotated)

            # Raw detections as JSON.
            with open(os.path.join(out_dir, f"{model['subdir']}.json"), "w") as f:
                json.dump(detections, f, indent=4)
        except Exception as error:
            image_errors += 1
            print(f"  [warn] {model['name']} failed on {os.path.basename(image_path)}: "
                  f"{str(error).splitlines()[0]}")
            continue

        peak_ram = max(peak_ram, process.memory_info().rss)
        current_vram = read_vram_mb(run_device)
        if current_vram is not None:
            peak_vram = current_vram if peak_vram is None else max(peak_vram, current_vram)

    # 4) Fill in the summary row.
    row["infer_s"] = round(total_infer, 3)
    row["detections"] = len(all_confidences)
    row["avg_conf"] = round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0
    row["categories"] = len(categories)
    row["ram_mb"] = round(peak_ram / (1024 * 1024), 1)
    row["vram_mb"] = round(peak_vram, 1) if peak_vram is not None else "N/A"

    status = "ok" if run_device == device else f"ok (fell back to {run_device})"
    if image_errors:
        status += f", {image_errors} image error(s)"
    row["status"] = status
    print(f"  [ ok ] {model['name']}: {row['detections']} detections across "
          f"{len(image_paths)} image(s), load {row['load_s']}s, infer {row['infer_s']}s")
    return row


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_table(rows):
    """Print an aligned summary table to the console."""
    headers = ["Model", "Load(s)", "Infer(s)", "Detections", "Avg Conf",
               "Categories", "RAM(MB)", "VRAM(MB)", "Size(MB)"]
    keys = ["model", "load_s", "infer_s", "detections", "avg_conf",
            "categories", "ram_mb", "vram_mb", "size_mb"]

    table = [headers] + [[str(row[k]) for k in keys] for row in rows]
    widths = [max(len(r[c]) for r in table) for c in range(len(headers))]

    def fmt(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("\n" + fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row_cells in table[1:]:
        print(fmt(row_cells))
    print()


def write_csv(rows):
    """Write the full benchmark results to a CSV file."""
    fields = ["model", "load_s", "infer_s", "detections", "avg_conf",
              "categories", "ram_mb", "vram_mb", "size_mb", "status"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main
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
    print(f"Found {len(image_paths)} image(s) in {IMAGES_DIR}/")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    rows = []
    for model in MODELS:
        print(f"\n=== {model['name']} ===")
        rows.append(benchmark_model(model, image_paths, device))

    print_table(rows)
    write_csv(rows)
    print(f"Annotated images and detections written to {OUTPUTS_DIR}/")
    print(f"Summary written to {CSV_PATH}")


if __name__ == "__main__":
    main()
