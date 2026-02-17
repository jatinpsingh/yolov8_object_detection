#!/usr/bin/env python3
"""
YOLOv8 Evaluation Script.
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
    parser = argparse.ArgumentParser(description="YOLOv8 Evaluation Script")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the trained model weights (e.g., best.pt)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to composites directory containing data.yaml")
    
    args = parser.parse_args()

    data_yaml = args.data_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: Dataset config not found at: {data_yaml}")
        sys.exit(1)

    if not args.model_path.exists():
        print(f"Error: Model weights not found at: {args.model_path}")
        sys.exit(1)

    print(f"Loading model: {args.model_path}")
    model = YOLO(str(args.model_path))

    # Determine experiment name from path (assuming .../experiment_name/weights/best.pt)
    # If structure differs, fallback to 'custom_model'
    if args.model_path.parent.name == "weights":
        experiment_name = args.model_path.parent.parent.name
    else:
        experiment_name = args.model_path.stem

    # Define base test directory: tests/<experiment_name>
    test_dir_base = PROJECT_ROOT / "tests" / experiment_name
    test_dir_base.mkdir(parents=True, exist_ok=True)

    # Find next run number: test_1, test_2, ...
    existing_tests = [d.name for d in test_dir_base.iterdir() if d.is_dir() and d.name.startswith("test_")]
    max_num = 0
    for test_name in existing_tests:
        try:
            num = int(test_name.split("_")[-1])
            if num > max_num:
                max_num = num
        except ValueError:
            continue
    
    run_name = f"test_{max_num + 1}"
    print(f"Saving evaluation results to: tests/{experiment_name}/{run_name}")

    print("Evaluating on test split...")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=1280,
        project=str(test_dir_base),
        name=run_name,
        verbose=True
    )
    
    print("EVALUATION SUMMARY")
    print(f"Model:         {args.model_path}")
    print("Split:         test")
    print(f"mAP50:         {metrics.box.map50:.4f}")
    print(f"mAP50-95:      {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()
