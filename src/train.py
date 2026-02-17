#!/usr/bin/env python3
"""
YOLOv8 Training Script.
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "composites"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to composites directory containing data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n", choices=["yolov8n", "yolov8s"], help="YOLOv8 variant: yolov8n or yolov8s")
    parser.add_argument("--retrain", action="store_true", help="Train entire model (all layers unfrozen). Default: Transfer learning.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--img-size", type=int, default=1280, help="Input image size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Run name for this experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    data_yaml = args.data_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: Dataset config not found at: {data_yaml}")
        print("Ensure you have generated the composites first.")
        sys.exit(1)

    if args.name:
        run_name = args.name
    else:
        mode_str = "retrain" if args.retrain else "transfer"
        run_name = f"{args.model}_{mode_str}"

    print(f"Loading pretrained model: {args.model}.pt")
    model = YOLO(f"{args.model}.pt")

    if args.retrain:
        print("Mode: Full Retrain (freeze=0)")
        freeze_value = 0
    else:
        print("Mode: Transfer Learning (freeze=10)")
        freeze_value = 10

    print(f"Starting training run: {run_name}")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        patience=args.patience,
        freeze=freeze_value,
        project=str(args.output),
        name=run_name,
        seed=args.seed,
        exist_ok=True,
        verbose=True
    )

    best_weights = args.output / run_name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights saved to: {best_weights}")

if __name__ == "__main__":
    main()
