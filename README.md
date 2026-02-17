# Object Detection & Localization Project

This project implements an object detection system capable of detecting, classifying, and locating multiple objects in a single composite image. It utilizes transfer learning with a pretrained YOLOv8 model.

## Project Overview

*   **Goal:** Build a multi-object detection system.
*   **Method:**
    1.  Generate synthetic composite images from single-object source images.
    2.  Fine-tune a pretrained YOLOv8 model on these composites.
    3.  Evaluate model performance (mAP, Precision, Recall).
    4.  Run inference on new images.
*   **Model:** YOLOv8 (Ultralytics).

## Setup

### Prerequisites

*   Anaconda or Miniconda installed.
*   Python 3.12 (recommended).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd object_detection
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n object_detection python=3.12 -y
    conda activate object_detection
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

```
object_detection/
├── composites/         # Generated synthetic dataset (images & labels)
├── configs/            # Configuration files (e.g., config.yaml)
├── data/               # Source single-object images
├── outputs/            # Training outputs and model checkpoints
├── src/                # Source code
│   ├── evaluate.py     # Evaluation script
│   ├── generate_composites.py # Script to generate synthetic data
│   ├── inference.py    # Inference script for detection
│   └── train.py        # Training script
├── tests/              # Inference results
├── requirements.txt    # Python dependencies
├── yolov8n.pt          # Pretrained YOLOv8 nano model
└── GEMINI.md           # Agent context (internal)
```

## Usage

### 1. Generate Synthetic Composites

Generate a dataset of composite images from the single-object source images in `data/`.

```bash
python src/generate_composites.py
```
*(Check the script for optional arguments like number of images, etc.)*

### 2. Train the Model

Fine-tune the YOLOv8 model on the generated composites.

```bash
python src/train.py
```
*(This will save the best model to `outputs/yolov8n_transfer/weights/best.pt`)*

### 3. Evaluate the Model

Evaluate the trained model on the test set.

```bash
python src/evaluate.py --model-path outputs/yolov8n_transfer/weights/best.pt
```

### 4. Run Inference

Detect objects in a specific image.

```bash
python src/inference.py <model_path> <image_path>
```

**Example:**
```bash
python src/inference.py outputs/yolov8n_transfer/weights/best.pt data/OBJ_001/001.jpg
```

The annotated image and detection results will be saved in the `tests/` directory (e.g., `tests/detection_1.jpg`, `tests/detection_1.txt`).

## Notes

*   **Data:** The `data/` directory contains folders for each object class (e.g., `OBJ_001`, `OBJ_002`).
*   **Configuration:** Adjust `configs/config.yaml` or `composites/data.yaml` to modify training parameters or dataset paths.
