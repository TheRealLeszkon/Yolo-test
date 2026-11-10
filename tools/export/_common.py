"""Shared helpers for the ONNX export scripts.

Kept intentionally dependency-light: only onnx / onnxruntime plus stdlib so the
export scripts remain reproducible and self-contained.
"""
from __future__ import annotations

import pathlib

# Repo root = two levels up from tools/export/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ONNX_OUT_DIR = REPO_ROOT / "models" / "onnx"


def onnx_out_dir() -> pathlib.Path:
    ONNX_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return ONNX_OUT_DIR


def validate_onnx(path: str | pathlib.Path) -> None:
    """Run onnx.checker on the file and confirm the IR is well formed."""
    import onnx

    path = pathlib.Path(path)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opset = ", ".join(
        f"{imp.domain or 'ai.onnx'}={imp.version}" for imp in model.opset_import
    )
    print(f"  [onnx.checker] OK  ->  {path.name}")
    print(f"  IR version : {model.ir_version}")
    print(f"  Opset      : {opset}")


def describe_onnx_io(path: str | pathlib.Path) -> None:
    """Print input/output names and shapes as recorded in the ONNX graph."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"  --- graph I/O for {pathlib.Path(path).name} ---")
    print("  Inputs:")
    for i in sess.get_inputs():
        print(f"    - {i.name:24s} shape={i.shape}  dtype={i.type}")
    print("  Outputs:")
    for o in sess.get_outputs():
        print(f"    - {o.name:24s} shape={o.shape}  dtype={o.type}")


def onnx_size_mb(path: str | pathlib.Path) -> float:
    return pathlib.Path(path).stat().st_size / (1024 * 1024)
