# Object Detection & Localization with YOLOv8

A YOLOv8 based object detection pipeline capable of detecting, classifying, and locating multiple objects in a single composite image. This project supports fully retraining YOLOv8 models as well as fine tuning pretrained YOLO models using transfer learning.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Workflow](#workflow)

## Project Overview

- **Goal:** Build a multi-object detection system.
- **Model:** YOLOv8 (Ultralytics).
- **Features:** 
    - Config-driven training (YAML).
    - Synthetic composite image generation from single-object source images
    - Automated train/val/test splitting.
    - Full re-training and Transfer Learning support.
    - Inference on new images.

## Environment Setup

This project uses Anaconda or Miniconda environment.

1. **Create and Activate Environment:**
   ```bash
   conda create -n object_detection python=3.12
   conda activate object_detection
   ```

2. **Install Dependencies:**
   Install the required libraries using `pip` and the provided `requirements.txt` file.
   ```bash
   pip install -r requirements.txt

## Project Structure

```
object_detection/
├── composites/                 # Generated synthetic dataset (images & labels)
├── configs/                    # Configuration files for composite dataset (e.g., config.yaml)
├── data/                       # Source single-object images
├── outputs/                    # Training outputs and model checkpoints
├── src/                        # Source code
│   ├── evaluate.py             # Evaluation script
│   ├── generate_composites.py  # Script to generate synthetic data
│   ├── inference.py            # Inference script for detection
│   └── train.py                # Training script
├── tests/                      # Inference results
├── requirements.txt            # Python dependencies
```
## Workflow

### 1. Dataset Download
Download the dataset from the following link:
[Google Drive Link](https://drive.google.com/drive/folders/1lQW22uf1tpphMuNlPRoQ8M4smt9w9qLB?usp=drive_link)

Extract/place the dataset folder (containing the 39 class subfolders) into the project directory so that it resides at `data/`.

### 2. Generate Synthetic Composites

Generate a dataset of composite images from the single-object source images in `data/`.

```bash
python src/generate_composites.py
```
*(Check the script for optional arguments)*

### 3. Train the Model

Fine-tune the YOLOv8 model on the generated composites.

```bash
python src/train.py
```
*(This will save the best model to `./outputs/yolov8n_transfer/weights/best.pt`)*
*(Check the script for optional arguments like yolo model, etc.)*

### 3. Evaluate the Model

Evaluate the trained model on the test set.

```bash
python src/evaluate.py --model-path <trained_model_path>
```
*(e.g. python src/evaluate.py --model-path outputs/yolov8n_transfer/weights/best.pt)*

### 4. Run Inference

Detect objects in a specific image.

```bash
python src/inference.py <model_path> <image_path>
```

The annotated image and detection results will be saved in the `tests/` directory (e.g., `tests/detection_1.jpg`, `tests/detection_1.txt`).

## Notes

*   **Data:** The `data/` directory contains folders for each object class (e.g., `OBJ_001`, `OBJ_002`).
*   **Configuration:** Adjust `configs/config.yaml` or `composites/data.yaml` to modify training parameters or dataset paths.
