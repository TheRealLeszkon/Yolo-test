#!/usr/bin/env bash
# Initialize Yolo-test: create the Python env and download model weights.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Syncing Python environment (uv)"
uv sync

echo "==> Downloading model weights"
dl() { # dl <url> <dest>
  if [ -f "$2" ]; then echo "  [skip] $2 already exists"; else
    echo "  [get ] $2"; curl -L --fail -o "$2" "$1"
  fi
}

# Segment Anything (SAM) checkpoints
dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  sam_vit_h_4b8939.pth
dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  sam_vit_l_0b3195.pth
dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  sam_vit_b_01ec64.pth
# SAM 2.1
dl https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt sam2.1_hiera_large.pt
# Ultralytics-hosted checkpoints
dl https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-x.pt      FastSAM-x.pt
dl https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt     mobile_sam.pt
dl https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt        yolo11n.pt
# EfficientSAM (via ultralytics assets)
dl https://github.com/ultralytics/assets/releases/download/v8.3.0/efficient_sam_vits.pt efficient_sam_vits.pt

echo "==> Note: models under ./models/ (yolo11, yoloworld, lvis, objects365) auto-download"
echo "    on first ultralytics run, or place them manually."
echo "==> Done."
