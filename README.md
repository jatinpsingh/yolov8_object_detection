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
├── runs/                       # Training/Val outputs and model checkpoints
├── src/                        # Source code
│   ├── generate_composites.py  # Script to generate synthetic data
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
yolo detect train model=yolov8s.pt data=composites/data.yaml imgsz=640 batch=16 epochs=100 freeze=10 project=outputs name=yolov8s_transfer
```
*(This will save the best model to `./runs/detect/outputs/yolov8n_transfer/weights/best.pt`)*
*(you can adjust parameters like model, imgsz, batch etc.)*

### 3. Evaluate the Model

Evaluate the trained model on the test set.

```bash
yolo detect val model=<path_to_trained_model> data=composites/data.yaml split=test
```

### 4. Run Inference

Detect objects in a specific image.

```bash
yolo detect predict model=<path_to_trained_model> source=<path_to_test_image>
```

The annotated image and detection results will be saved in the `runs/detect/` directory.

## Notes

*   **Data:** The `data/` directory contains folders for each object class (e.g., `OBJ_001`, `OBJ_002`).
*   **Configuration:** Adjust `configs/config.yaml` or `composites/data.yaml` to modify training parameters or dataset paths.
