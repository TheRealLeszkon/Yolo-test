# Yolo-test

Benchmarks for choosing an on-device detection and segmentation pipeline.

- `benchmark.py`: times SAM variants (SAM ViT-B/L/H, MobileSAM, FastSAM, EfficientSAM, SAM2) on the same images.
- `yolo_sam_benchmark.py`: detector boxes (YOLOE with Objects365 / LVIS vocabularies, YOLO-World) used as prompts for FastSAM and EfficientSAM.
- `filter_benchmark.py`: a geometry-only filter that removes tiny boxes and boxes nested inside another (for example a drawer inside a table).
- `tools/export/`: exports YOLOE and EfficientSAM to ONNX for ONNX Runtime Mobile. See its README.

## Result

EfficientSAM took about **12.5 ms per mask** against FastSAM's **456 ms**, roughly 36 times faster, across 4 scenes and 3 detectors (`benchmark_results.csv`). It encodes the image once and reuses that for every box.

Model weights are not included; `./initialize.sh` sets up the environment and downloads them.
