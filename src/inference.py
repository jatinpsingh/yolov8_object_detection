#!/usr/bin/env python3
"""
YOLOv8 Inference Script.
"""

import argparse
import sys
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_next_detection_number(output_dir: Path) -> int:
    """
    Finds the next available detection number in the output directory.
    Checks for files matching 'detection_<number>.*'.
    """
    max_num = 0
    if not output_dir.exists():
        return 1
    
    for file_path in output_dir.glob("detection_*.*"):
        try:
            # Extract number from filename (e.g., detection_1.jpg -> 1)
            stem = file_path.stem # detection_1
            parts = stem.split("_")
            if len(parts) >= 2 and parts[0] == "detection":
                num = int(parts[1])
                if num > max_num:
                    max_num = num
        except ValueError:
            continue
    return max_num + 1

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    parser.add_argument("model_path", type=Path, help="Path to the trained model weights (e.g., best.pt)")
    parser.add_argument("image_path", type=Path, help="Path to the image to run inference on")
    
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model weights not found at: {args.model_path}")
        sys.exit(1)

    if not args.image_path.exists():
        print(f"Error: Image not found at: {args.image_path}")
        sys.exit(1)

    # Output directory
    output_dir = PROJECT_ROOT / "tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model = YOLO(str(args.model_path))

    print(f"Running inference on: {args.image_path}")
    results = model(str(args.image_path))

    # Process results (assuming single image)
    result = results[0]
    
    # Get next detection number
    next_num = get_next_detection_number(output_dir)
    base_filename = f"detection_{next_num}"
    
    # Save annotated image
    annotated_frame = result.plot()
    output_image_path = output_dir / f"{base_filename}.jpg"
    cv2.imwrite(str(output_image_path), annotated_frame)
    print(f"Saved annotated image to: {output_image_path}")

    # Extract and save object names
    detected_indices = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names
    detected_names = [class_names[idx] for idx in detected_indices]
    
    output_text_path = output_dir / f"{base_filename}.txt"
    with open(output_text_path, "w") as f:
        for name in detected_names:
            f.write(f"{name}\n")
    
    print(f"Saved detected object names to: {output_text_path}")
    print("Detected objects:")
    for name in detected_names:
        print(f"- {name}")

if __name__ == "__main__":
    main()
