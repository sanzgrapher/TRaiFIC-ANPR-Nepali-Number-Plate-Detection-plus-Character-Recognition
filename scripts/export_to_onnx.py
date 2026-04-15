#!/usr/bin/env python3
"""
One-time script to export the three PyTorch model files to ONNX format.

Run this locally (where ultralytics and torch are installed) **before** the
first Vercel deployment.  The resulting .onnx files must be committed to the
repository alongside the original .pt / .pth files.

Usage
-----
    # From the repo root:
    pip install "ultralytics>=8.3" torch
    python scripts/export_to_onnx.py

The script writes:
    application/models/pd_traific_v2_mix.onnx
    application/models/sg_traific_v12.onnx
    application/models/char_traific_v3.onnx
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "application", "models")
APP_DIR = os.path.join(REPO_ROOT, "application")


# ---------------------------------------------------------------------------
# YOLO models  (.pt  →  .onnx via ultralytics)
# ---------------------------------------------------------------------------

def export_yolo(pt_path: str, onnx_path: str) -> None:
    """Export a YOLO .pt model to ONNX format."""
    from ultralytics import YOLO  # type: ignore

    print(f"Exporting YOLO model: {pt_path}")
    model = YOLO(pt_path)
    # ultralytics writes the .onnx next to the .pt file with the same stem
    result = model.export(format="onnx", imgsz=640, simplify=True, opset=12)
    exported = str(result)  # ultralytics returns the output path

    if os.path.abspath(exported) != os.path.abspath(onnx_path):
        os.replace(exported, onnx_path)

    print(f"  ✓  saved to {onnx_path}  ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Custom CNN  (.pth  →  .onnx via torch.onnx.export)
# ---------------------------------------------------------------------------

def export_cnn(pth_path: str, onnx_path: str, num_classes: int) -> None:
    """Export the NepaliPlateCNN .pth checkpoint to ONNX format."""
    import torch  # type: ignore

    sys.path.insert(0, APP_DIR)
    from models import NepaliPlateCNN  # type: ignore

    print(f"Exporting CNN model: {pth_path}")
    device = torch.device("cpu")
    model = NepaliPlateCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    dummy_input = torch.zeros(1, 1, 32, 32)  # [batch, channels, H, W]
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"  ✓  saved to {onnx_path}  ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Read NUM_CLASSES from config without importing the full application stack
    sys.path.insert(0, APP_DIR)
    import config  # type: ignore

    export_yolo(
        os.path.join(MODELS_DIR, "pd_traific_v2_mix.pt"),
        os.path.join(MODELS_DIR, "pd_traific_v2_mix.onnx"),
    )
    export_yolo(
        os.path.join(MODELS_DIR, "sg_traific_v12.pt"),
        os.path.join(MODELS_DIR, "sg_traific_v12.onnx"),
    )
    export_cnn(
        os.path.join(MODELS_DIR, "char_traific_v3.pth"),
        os.path.join(MODELS_DIR, "char_traific_v3.onnx"),
        num_classes=config.NUM_CLASSES,
    )

    print("\nAll models exported successfully.")
    print("Commit the new .onnx files to the repository before deploying.")
