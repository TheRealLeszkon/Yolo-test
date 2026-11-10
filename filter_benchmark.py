"""
filter_benchmark.py
===================

Benchmark simple, geometry-only post-processing that reduces the number of
segmented objects coming out of the YOLO -> EfficientSAM pipeline, before they
would be handed to an LLM. No AI / LLM logic is used here -- only geometry.

Two filtering stages are applied and measured:

    Stage 1  Geometry filtering    -- drop tiny / sliver / low-fill segments.
    Stage 2  Parent-child filtering -- drop a segment that is almost entirely
             contained inside a larger segment (e.g. a drawer inside a table).

This script does NOT modify or rerun the previous benchmark. It consumes the
cutout PNGs it produced under:

    outputs/<image>/segmentations/<detector>_efficientsam/<NN>_<label>.png

The previous benchmark saved each object as a tightly-cropped, transparent-
background cutout but did not store per-object bounding boxes / global mask
positions. Because a cutout's RGB is an exact rectangular crop of the original
image, we recover each object's position by locating that crop in the original
(exact template match) and pasting the alpha mask back. This gives us the
full-image mask, bounding box, label and area for every object -- without
rerunning YOLO or SAM. (Per-object confidence was not saved, so the confidence
threshold is only applied when a confidence value is available.)

    python filter_benchmark.py
"""

import csv
import glob
import os
import shutil
import time

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Thresholds -- tweak these to change how aggressive each stage is.
# -----------------------------------------------------------------------------
# Stage 1 -- geometry filtering
MIN_MASK_AREA_PX = 500        # drop masks smaller than this many pixels
MIN_BOX_AREA_PX = 900         # drop bounding boxes smaller than this many pixels
MIN_CONFIDENCE = 0.0          # drop detections below this confidence (if available)
MIN_MASK_TO_BOX_RATIO = 0.10  # mask must fill at least this fraction of its box
MAX_ASPECT_RATIO = 8.0        # drop thin slivers (longer_side / shorter_side)

# Stage 2 -- parent-child filtering
MIN_CONTAINMENT = 0.90        # child must be >= this fraction inside the parent
MAX_CHILD_AREA_RATIO = 0.5    # child must be < this fraction of the parent's area

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
OUTPUTS_DIR = "outputs"          # produced by the previous benchmark
IMAGES_DIR = "images"            # original images
FILTERED_DIR = "filtered_outputs"
CSV_PATH = "filter_benchmark.csv"

# We post-process the EfficientSAM segmentations specifically.
SEG_SUFFIX = "_efficientsam"


# -----------------------------------------------------------------------------
# Loading objects back from the previous benchmark's cutouts
# -----------------------------------------------------------------------------
def find_original_image(image_name):
    """Return the path to the original image whose stem is ``image_name``."""
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(IMAGES_DIR, image_name + ext)
        if os.path.exists(path):
            return path
    return None


def locate_crop(original_bgr, crop_bgr):
    """Return the (x, y) offset of ``crop_bgr`` inside ``original_bgr``.

    The cutout's colour channels are an exact rectangular crop of the original,
    so template matching finds the position exactly. Returns None if the crop
    does not fit or cannot be matched exactly.
    """
    oh, ow = original_bgr.shape[:2]
    ch, cw = crop_bgr.shape[:2]
    if ch > oh or cw > ow:
        return None
    result = cv2.matchTemplate(original_bgr, crop_bgr, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    if np.array_equal(original_bgr[y:y + ch, x:x + cw], crop_bgr):
        return x, y
    return x, y  # best effort; near-duplicate regions are extremely unlikely


def load_objects(seg_dir, original_bgr):
    """Load every cutout in ``seg_dir`` as a full-image object record.

    Each record has: label, mask (full-image bool), bbox [x1,y1,x2,y2],
    mask_area, box_area, confidence (None -- not saved by the previous run),
    and the cutout path (so removed objects can be copied out later).
    """
    height, width = original_bgr.shape[:2]
    objects = []
    for path in sorted(glob.glob(os.path.join(seg_dir, "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        label = name.split("_", 1)[1].replace("_", " ") if "_" in name else name

        cutout = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if cutout is None or cutout.ndim != 3 or cutout.shape[2] != 4:
            continue
        crop_bgr = cutout[:, :, :3]
        local_mask = cutout[:, :, 3] > 127
        ch, cw = local_mask.shape

        offset = locate_crop(original_bgr, crop_bgr)
        if offset is None:
            print(f"    [warn] could not place {os.path.basename(path)}, skipping")
            continue
        x, y = offset

        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y:y + ch, x:x + cw] = local_mask

        objects.append({
            "label": label,
            "path": path,
            "mask": full_mask,
            "bbox": [x, y, x + cw, y + ch],
            "mask_area": int(local_mask.sum()),
            "box_area": int(cw * ch),
            "confidence": None,
        })
    return objects


# -----------------------------------------------------------------------------
# Stage 1 -- geometry filtering
# -----------------------------------------------------------------------------
def geometry_filter(objects):
    """Split ``objects`` into (kept, removed). Removed items carry a 'reason'."""
    kept, removed = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        box_w, box_h = x2 - x1, y2 - y1
        fill_ratio = obj["mask_area"] / obj["box_area"] if obj["box_area"] else 0.0
        aspect = max(box_w, box_h) / max(min(box_w, box_h), 1)

        reasons = []
        if obj["mask_area"] < MIN_MASK_AREA_PX:
            reasons.append(f"mask area {obj['mask_area']}px < {MIN_MASK_AREA_PX}")
        if obj["box_area"] < MIN_BOX_AREA_PX:
            reasons.append(f"box area {obj['box_area']}px < {MIN_BOX_AREA_PX}")
        if obj["confidence"] is not None and obj["confidence"] < MIN_CONFIDENCE:
            reasons.append(f"confidence {obj['confidence']:.2f} < {MIN_CONFIDENCE}")
        if fill_ratio < MIN_MASK_TO_BOX_RATIO:
            reasons.append(f"mask/box fill {fill_ratio:.2f} < {MIN_MASK_TO_BOX_RATIO}")
        if aspect > MAX_ASPECT_RATIO:
            reasons.append(f"aspect ratio {aspect:.1f} > {MAX_ASPECT_RATIO}")

        if reasons:
            removed.append({**obj, "reason": "; ".join(reasons)})
        else:
            kept.append(obj)
    return kept, removed


# -----------------------------------------------------------------------------
# Stage 2 -- parent-child filtering
# -----------------------------------------------------------------------------
def parent_child_filter(objects):
    """Remove objects that are mostly contained inside a larger object.

    Purely geometric: for each object we check whether it is at least
    MIN_CONTAINMENT contained in some larger object whose area is big enough
    that this object is < MAX_CHILD_AREA_RATIO of it. Object names are never
    used. Returns (kept, removed); removed items carry parent/containment info.
    """
    kept, removed = [], []
    for i, child in enumerate(objects):
        child_area = child["mask_area"]
        best_parent = None
        best_containment = 0.0

        for j, parent in enumerate(objects):
            if i == j:
                continue
            if child_area >= parent["mask_area"] * MAX_CHILD_AREA_RATIO:
                continue  # not significantly smaller than this candidate parent
            intersection = int(np.logical_and(child["mask"], parent["mask"]).sum())
            containment = intersection / child_area if child_area else 0.0
            if containment >= MIN_CONTAINMENT and containment > best_containment:
                best_containment = containment
                best_parent = parent

        if best_parent is not None:
            union = int(np.logical_or(child["mask"], best_parent["mask"]).sum())
            intersection = int(np.logical_and(child["mask"], best_parent["mask"]).sum())
            iou = intersection / union if union else 0.0
            area_ratio = child_area / best_parent["mask_area"]
            reason = (f"Contained {best_containment * 100:.1f}% inside "
                      f"{best_parent['label']}\nArea ratio {area_ratio:.2f}\n"
                      f"IoU {iou:.2f}")
            removed.append({**child, "reason": reason, "parent": best_parent["label"],
                            "containment": round(best_containment, 4),
                            "area_ratio": round(area_ratio, 4), "iou": round(iou, 4)})
        else:
            kept.append(child)
    return kept, removed


# -----------------------------------------------------------------------------
# Visualisation / saving
# -----------------------------------------------------------------------------
def draw_objects(image_bgr, objects):
    """Return a copy of the image with each object's box and label drawn."""
    canvas = image_bgr.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        text = obj["label"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 4)
        cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 4, ty), (0, 220, 0), -1)
        cv2.putText(canvas, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_removed(folder, removed):
    """Copy each removed object's cutout into ``folder`` with a reason .txt."""
    os.makedirs(folder, exist_ok=True)
    for obj in removed:
        stem = os.path.splitext(os.path.basename(obj["path"]))[0]
        shutil.copy(obj["path"], os.path.join(folder, stem + ".png"))
        with open(os.path.join(folder, stem + ".txt"), "w") as f:
            f.write("Reason:\n" + obj["reason"] + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Find every <detector>_efficientsam folder produced by the previous run.
    combos = []
    for seg_dir in sorted(glob.glob(os.path.join(
            OUTPUTS_DIR, "*", "segmentations", "*" + SEG_SUFFIX))):
        image_name = seg_dir.split(os.sep)[-3]
        detector = os.path.basename(seg_dir)[:-len(SEG_SUFFIX)]
        combos.append((image_name, detector, seg_dir))

    if not combos:
        print(f"No '*{SEG_SUFFIX}' segmentation folders found under {OUTPUTS_DIR}/.")
        print("Run the YOLO -> EfficientSAM benchmark first.")
        return

    print(f"Found {len(combos)} segmentation set(s) to filter.\n")
    os.makedirs(FILTERED_DIR, exist_ok=True)

    rows = []
    total_geometry_time = 0.0
    total_parent_child_time = 0.0

    for image_name, detector, seg_dir in combos:
        image_path = find_original_image(image_name)
        if image_path is None:
            print(f"[warn] original image for '{image_name}' not found, skipping")
            continue
        original_bgr = cv2.imread(image_path)

        objects = load_objects(seg_dir, original_bgr)
        original_count = len(objects)

        start = time.perf_counter()
        after_geometry, removed_geometry = geometry_filter(objects)
        geometry_time = time.perf_counter() - start

        start = time.perf_counter()
        after_parent_child, removed_parent_child = parent_child_filter(after_geometry)
        parent_child_time = time.perf_counter() - start

        total_geometry_time += geometry_time
        total_parent_child_time += parent_child_time

        # --- write the visual outputs for this (image, detector) ---
        out_dir = os.path.join(FILTERED_DIR, image_name, detector + SEG_SUFFIX)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "stage0_original.png"),
                    draw_objects(original_bgr, objects))
        cv2.imwrite(os.path.join(out_dir, "stage1_geometry.png"),
                    draw_objects(original_bgr, after_geometry))
        cv2.imwrite(os.path.join(out_dir, "stage2_parent_child.png"),
                    draw_objects(original_bgr, after_parent_child))
        save_removed(os.path.join(out_dir, "removed_geometry"), removed_geometry)
        save_removed(os.path.join(out_dir, "removed_parent_child"), removed_parent_child)

        rows.append({
            "image": image_name,
            "detector": detector,
            "original": original_count,
            "after_geometry": len(after_geometry),
            "after_parent_child": len(after_parent_child),
            "geometry_removed": len(removed_geometry),
            "parent_child_removed": len(removed_parent_child),
            "geometry_time_s": round(geometry_time, 4),
            "parent_child_time_s": round(parent_child_time, 4),
        })
        print(f"  {image_name} / {detector}: {original_count} -> "
              f"{len(after_geometry)} (geometry) -> "
              f"{len(after_parent_child)} (parent-child)")

    print_summary(rows)
    print_statistics(rows, total_geometry_time, total_parent_child_time)
    write_csv(rows)
    print(f"\nVisual outputs written to {FILTERED_DIR}/")
    print(f"Report written to {CSV_PATH}")


def print_summary(rows):
    """Print the per-(image, detector) summary table."""
    headers = ["Image", "Detector", "Original", "Geometry", "ParentChild", "Final"]
    table = [headers]
    for r in rows:
        table.append([r["image"], r["detector"], str(r["original"]),
                      str(r["after_geometry"]), str(r["after_parent_child"]),
                      str(r["after_parent_child"])])
    widths = [max(len(row[c]) for row in table) for c in range(len(headers))]
    fmt = lambda cells: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    print("\n" + fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in table[1:]:
        print(fmt(row))


def print_statistics(rows, total_geometry_time, total_parent_child_time):
    """Print aggregate statistics across every processed set."""
    if not rows:
        return
    geometry_pcts = [100 * r["geometry_removed"] / r["original"]
                     for r in rows if r["original"]]
    parent_pcts = [100 * r["parent_child_removed"] / r["after_geometry"]
                   for r in rows if r["after_geometry"]]
    avg_final = sum(r["after_parent_child"] for r in rows) / len(rows)

    print("\nStatistics")
    print("-" * 40)
    print(f"Avg removed by geometry     : {sum(geometry_pcts) / len(geometry_pcts):.1f}%")
    print(f"Avg removed by parent-child : {sum(parent_pcts) / len(parent_pcts):.1f}%")
    print(f"Avg final object count      : {avg_final:.1f}")
    print(f"Total geometry stage time   : {total_geometry_time:.4f}s")
    print(f"Total parent-child time     : {total_parent_child_time:.4f}s")


def write_csv(rows):
    """Write the per-(image, detector) report to CSV."""
    fields = ["image", "detector", "original", "after_geometry", "after_parent_child",
              "geometry_removed", "parent_child_removed", "geometry_time_s",
              "parent_child_time_s"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
