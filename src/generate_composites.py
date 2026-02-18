import argparse
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "composites"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Global shared dataset for workers
# Structure: {'train': {class_id: [paths]}, 'val': {...}, 'test': {...}}
SHARED_DATASET = {}

# Supported background types
BG_TYPES = ["noise", "solid", "white", "black", "gradient"]


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if "batches" not in config:
        raise ValueError("Config missing 'batches' key")
    
    def validate_batch_list(batch_list, section_name):
        seen_names = set()
        for idx, batch in enumerate(batch_list):
            if "name" not in batch:
                raise ValueError(f"Batch {idx} in {section_name} missing 'name'")
            if "obj_range" not in batch or not isinstance(batch["obj_range"], list) or len(batch["obj_range"]) != 2:
                raise ValueError(f"Batch {batch['name']} invalid 'obj_range'")
            if batch["obj_range"][0] < 0 or batch["obj_range"][1] < batch["obj_range"][0]:
                raise ValueError(f"Batch {batch['name']} invalid 'obj_range' values")
            if "obj_size" not in batch or batch["obj_size"] <= 0:
                raise ValueError(f"Batch {batch['name']} invalid 'obj_size'")
            
            # Validate bg_type if specified
            bg_type = batch.get("bg_type", "noise")
            if bg_type not in BG_TYPES and bg_type != "random":
                raise ValueError(
                    f"Batch {batch['name']} invalid 'bg_type': {bg_type}. "
                    f"Must be one of {BG_TYPES + ['random']}"
                )
            
            # Validate canvas_size if specified
            canvas_size = batch.get("canvas_size", None)
            if canvas_size is not None and canvas_size <= 0:
                raise ValueError(f"Batch {batch['name']} invalid 'canvas_size': must be > 0")
            
            # Warn if obj_size exceeds canvas_size (will be clamped at generation time)
            if canvas_size is not None and batch["obj_size"] > canvas_size * 0.85:
                print(
                    f"Warning: Batch '{batch['name']}' obj_size ({batch['obj_size']}) is large "
                    f"relative to canvas_size ({canvas_size}). Will be clamped during generation."
                )
            
            # Validate count if specified
            count = batch.get("count", None)
            if count is not None and count <= 0:
                raise ValueError(f"Batch {batch['name']} invalid 'count': must be > 0")
            
            if batch["name"] in seen_names:
                raise ValueError(f"Duplicate batch name '{batch['name']}' in {section_name}")
            seen_names.add(batch["name"])

    validate_batch_list(config.get("batches", []), "batches")
    if "test_batches" in config:
        validate_batch_list(config["test_batches"], "test_batches")
        
    return config


def load_dataset(data_dir: Path) -> Tuple[Dict[int, List[Path]], List[str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    dataset = {}
    class_names = []

    for class_id, class_dir in enumerate(subdirs):
        class_names.append(class_dir.name)
        images = sorted(
            list(class_dir.glob("*.jpg")) + 
            list(class_dir.glob("*.jpeg")) + 
            list(class_dir.glob("*.png"))
        )
        if images:
            dataset[class_id] = images
    
    print(f"Loaded {sum(len(v) for v in dataset.values())} images from {len(dataset)} classes.")
    return dataset, class_names


def split_source_dataset(dataset: Dict[int, List[Path]], seed: int) -> Dict[str, Dict[int, List[Path]]]:
    """Splits source images into train (80%), val (10%), test (10%) pools to prevent leakage."""
    rng = random.Random(seed)
    splits = {"train": {}, "val": {}, "test": {}}
    
    for class_id, images in dataset.items():
        imgs = list(images)
        rng.shuffle(imgs)
        
        n = len(imgs)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        # Ensure at least 1 image per split if possible
        if n_train == 0 and n > 0:
            n_train = 1
        
        splits["train"][class_id] = imgs[:n_train]
        splits["val"][class_id] = imgs[n_train:n_train + n_val]
        splits["test"][class_id] = imgs[n_train + n_val:]
        
    return splits


def init_worker(shared_data):
    global SHARED_DATASET
    SHARED_DATASET = shared_data


def create_background(size: int, seed: int, bg_type: str = "noise") -> Image.Image:
    """Creates a background image of the given type.
    
    Supported types:
        noise    - Random color base + gaussian noise (original behavior)
        solid    - Random solid color
        white    - Pure white
        black    - Pure black
        gradient - Linear gradient between two random colors
        random   - Randomly picks one of the above
    """
    rng = np.random.default_rng(seed)
    
    if bg_type == "random":
        bg_type = rng.choice(BG_TYPES)
    
    if bg_type == "white":
        return Image.new("RGB", (size, size), (255, 255, 255))
    
    elif bg_type == "black":
        return Image.new("RGB", (size, size), (0, 0, 0))
    
    elif bg_type == "solid":
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        return Image.new("RGB", (size, size), color)
    
    elif bg_type == "gradient":
        color1 = rng.integers(0, 256, size=3).astype(np.float64)
        color2 = rng.integers(0, 256, size=3).astype(np.float64)
        # Vectorized gradient (vertical or horizontal)
        t = np.linspace(0, 1, size).reshape(-1, 1)
        gradient_1d = (color1 * (1 - t) + color2 * t).astype(np.uint8)  # (size, 3)
        if rng.random() > 0.5:
            # Vertical gradient: broadcast across columns
            arr = np.broadcast_to(gradient_1d[:, np.newaxis, :], (size, size, 3)).copy()
        else:
            # Horizontal gradient: broadcast across rows
            arr = np.broadcast_to(gradient_1d[np.newaxis, :, :], (size, size, 3)).copy()
        return Image.fromarray(arr)
    
    else:  # "noise" (default / original behavior)
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        img = Image.new("RGB", (size, size), color)
        img_arr = np.array(img).astype(np.float32)
        noise = rng.normal(0, 25, img_arr.shape).astype(np.float32)
        img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr)


def get_tight_bbox(img: Image.Image) -> Tuple[int, int, int, int]:
    try:
        alpha = np.array(img.split()[3])
    except IndexError:
        return 0, 0, img.width, img.height

    non_transparent = np.where(alpha > 0)
    
    if non_transparent[0].size == 0 or non_transparent[1].size == 0:
        return 0, 0, 0, 0
    
    min_y, max_y = np.min(non_transparent[0]), np.max(non_transparent[0])
    min_x, max_x = np.min(non_transparent[1]), np.max(non_transparent[1])
    
    return min_x, min_y, (max_x - min_x + 1), (max_y - min_y + 1)


def augment_object(
    img_path: Path, obj_size: int, canvas_size: int,
    is_train: bool, rng: np.random.Generator
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Load, resize, and augment a single object image.
    
    The effective obj_size is clamped to 85% of canvas_size to leave room
    for rotation expansion and placement margin.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGBA")
        
        # Clamp obj_size to prevent objects exceeding canvas
        max_obj_size = int(canvas_size * 0.85)
        effective_size = min(obj_size, max_obj_size)
        
        if is_train:
            scale_factor = rng.uniform(0.7, 1.3)
        else:
            scale_factor = 1.0
            
        target_size = int(effective_size * scale_factor)
        # Safety clamp after scaling
        target_size = min(target_size, max_obj_size)
        target_size = max(target_size, 16)  # minimum viable size
        
        # Resize preserving aspect ratio
        w, h = img.size
        aspect = w / h
        if aspect > 1:
            new_w = target_size
            new_h = max(1, int(target_size / aspect))
        else:
            new_h = target_size
            new_w = max(1, int(target_size * aspect))
            
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        if not is_train:
            return img, get_tight_bbox(img)

        if rng.random() < 0.5:
            img = ImageOps.mirror(img)
            
        angle = rng.uniform(-30, 30)
        img = img.rotate(
            angle, resample=Image.Resampling.BICUBIC,
            expand=True, fillcolor=(0, 0, 0, 0)
        )
        
        # Downscale if rotation expansion made it too large for canvas
        if img.width > canvas_size or img.height > canvas_size:
            downscale = min(canvas_size / img.width, canvas_size / img.height) * 0.95
            img = img.resize(
                (max(1, int(img.width * downscale)), max(1, int(img.height * downscale))),
                Image.Resampling.BILINEAR
            )
        
        r, g, b, a = img.split()
        rgb_img = Image.merge("RGB", (r, g, b))

        enhancer = ImageEnhance.Brightness(rgb_img)
        rgb_img = enhancer.enhance(rng.uniform(0.7, 1.3))
        
        enhancer = ImageEnhance.Contrast(rgb_img)
        rgb_img = enhancer.enhance(rng.uniform(0.7, 1.3))
        
        if rng.random() < 0.3:
            radius = rng.uniform(0.5, 1.5)
            rgb_img = rgb_img.filter(ImageFilter.GaussianBlur(radius))
            
        r, g, b = rgb_img.split()
        img = Image.merge("RGBA", (r, g, b, a))
            
        bbox = get_tight_bbox(img)
        
        return img, bbox


def check_iou(
    box1: Tuple[float, float, float, float],
    boxes: List[Tuple[float, float, float, float]]
) -> bool:
    x1, y1, w1, h1 = box1
    box1_area = w1 * h1
    if box1_area <= 0:
        return False

    for b in boxes:
        x2, y2, w2, h2 = b
        box2_area = w2 * h2
        if box2_area <= 0:
            continue

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        intersection = inter_w * inter_h
        
        union = box1_area + box2_area - intersection
        
        if union > 0 and (intersection / union) > 0.10:
            return True
            
    return False


def generate_single_composite(task_args) -> Dict:
    (global_idx, batch_name, obj_range, obj_size, canvas_size, is_train, 
     output_dirs, seed, split_name, bg_type) = task_args
    
    random.seed(seed)
    rng = np.random.default_rng(seed)
    
    background = create_background(canvas_size, seed, bg_type)
    
    num_objects = random.randint(obj_range[0], obj_range[1])
    
    placed_boxes = []
    labels = []
    
    # Use the appropriate split of source images
    source_pool = SHARED_DATASET.get(split_name, {})
    available_classes = list(source_pool.keys())
    
    if not available_classes:
        return {"success": False, "reason": f"No classes found for split {split_name}"}

    for _ in range(num_objects):
        class_id = random.choice(available_classes)
        images = source_pool[class_id]
        if not images:
            continue
        
        img_path = random.choice(images)
        
        img_obj, bbox_rel = augment_object(img_path, obj_size, canvas_size, is_train, rng)
        
        bx, by, bw, bh = bbox_rel
        if bw <= 0 or bh <= 0:
            continue
            
        placed = False
        for _ in range(100):
            max_x = max(0, canvas_size - img_obj.width)
            max_y = max(0, canvas_size - img_obj.height)
            
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)
            
            abs_box = (paste_x + bx, paste_y + by, bw, bh)
            
            if not check_iou(abs_box, placed_boxes):
                background.paste(img_obj, (paste_x, paste_y), img_obj)
                
                x_center = (abs_box[0] + abs_box[2] / 2) / canvas_size
                y_center = (abs_box[1] + abs_box[3] / 2) / canvas_size
                w_norm = abs_box[2] / canvas_size
                h_norm = abs_box[3] / canvas_size
                
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                placed_boxes.append(abs_box)
                placed = True
                break
    
    filename = f"{batch_name}_{global_idx:06d}"
    
    img_out_path = output_dirs["images"] / split_name / f"{filename}.jpg"
    lbl_out_path = output_dirs["labels"] / split_name / f"{filename}.txt"
    
    background.save(img_out_path, "JPEG", quality=75)
    
    with open(lbl_out_path, "w") as f:
        f.write("\n".join(labels))
        
    return {"success": True, "num_objects": len(labels), "batch": batch_name}


def main():
    parser = argparse.ArgumentParser(description="Synthetic Composite Image Generator")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Source data directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config yaml")
    parser.add_argument("--canvas-size", type=int, default=1280, help="Default canvas size (pixels)")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/Val split ratio")
    parser.add_argument("--batch-size", type=int, default=500, help="Default images per train/val batch")
    parser.add_argument("--test-size", type=int, default=50, help="Default images per test batch")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    if args.output.exists():
        print(f"Output directory {args.output} exists. Removing...")
        shutil.rmtree(args.output)
    
    for split in ["train", "val", "test"]:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    
    full_dataset, class_names = load_dataset(args.data_dir)
    if not full_dataset:
        print("Error: No images found in data directory!")
        sys.exit(1)
        
    # Split source images to prevent data leakage
    print("Splitting source images into Train (80%), Val (10%), Test (10%)...")
    split_dataset_map = split_source_dataset(full_dataset, args.seed)
    
    # Verify splits
    total_imgs = sum(len(v) for v in full_dataset.values())
    train_imgs = sum(len(v) for v in split_dataset_map["train"].values())
    val_imgs = sum(len(v) for v in split_dataset_map["val"].values())
    test_imgs = sum(len(v) for v in split_dataset_map["test"].values())
    print(f"Source split: Train={train_imgs}, Val={val_imgs}, Test={test_imgs} (Total={total_imgs})")

    # --- Build train/val tasks ---
    tasks = []
    random.seed(args.seed)
    
    batches = config.get("batches", [])
    if not batches:
        print("Warning: No train/val batches defined in config.")

    # Compute total train/val images (sum of per-batch counts)
    batch_counts = []
    for batch in batches:
        count = batch.get("count", args.batch_size)
        batch_counts.append(count)
    total_tv_images = sum(batch_counts)
    
    # Determine train/val split indices across all batches
    indices = list(range(total_tv_images))
    random.shuffle(indices)
    split_idx = int(total_tv_images * args.train_ratio)
    train_indices = set(indices[:split_idx])
    
    global_counter = 0
    print("\nTrain/Val batch plan:")
    for batch, count in zip(batches, batch_counts):
        canvas_size = batch.get("canvas_size", args.canvas_size)
        bg_type = batch.get("bg_type", "noise")
        print(
            f"  {batch['name']:<25} | count={count:<5} | canvas={canvas_size:<5} "
            f"| obj_size={batch['obj_size']:<4} | obj_range={batch['obj_range']} "
            f"| bg={bg_type}"
        )
        for _ in range(count):
            split = "train" if global_counter in train_indices else "val"
            is_train = (split == "train")
            
            task = (
                global_counter, batch["name"], batch["obj_range"], batch["obj_size"],
                canvas_size, is_train,
                {"images": args.output / "images", "labels": args.output / "labels"},
                args.seed + global_counter, split, bg_type
            )
            tasks.append(task)
            global_counter += 1

    # --- Build test tasks ---
    test_batches = config.get("test_batches", [])
    test_counter = 0
    seen_test_names = set()
    
    print("\nTest batch plan:")
    for batch in test_batches:
        b_name = batch["name"]
        if not b_name.startswith("test_"):
            b_name = f"test_{b_name}"
        
        if b_name in seen_test_names:
            print(f"Error: Duplicate test batch name after prefix resolution: '{b_name}'. Fix config.yaml.")
            sys.exit(1)
        seen_test_names.add(b_name)
        
        count = batch.get("count", args.test_size)
        canvas_size = batch.get("canvas_size", args.canvas_size)
        bg_type = batch.get("bg_type", "noise")
        print(
            f"  {b_name:<25} | count={count:<5} | canvas={canvas_size:<5} "
            f"| obj_size={batch['obj_size']:<4} | obj_range={batch['obj_range']} "
            f"| bg={bg_type}"
        )
        for _ in range(count):
            task = (
                test_counter, b_name, batch["obj_range"], batch["obj_size"],
                canvas_size, False,
                {"images": args.output / "images", "labels": args.output / "labels"},
                args.seed + total_tv_images + test_counter, "test", bg_type
            )
            tasks.append(task)
            test_counter += 1

    n_train = len([t for t in tasks if t[8] == "train"])
    n_val = len([t for t in tasks if t[8] == "val"])
    n_test = len([t for t in tasks if t[8] == "test"])
    print(f"\nTotal tasks: {len(tasks)} (train={n_train}, val={n_val}, test={n_test})")

    # --- Generate ---
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=init_worker, initargs=(split_dataset_map,)
    ) as executor:
        if tqdm:
            results = list(tqdm(
                executor.map(generate_single_composite, tasks),
                total=len(tasks), unit="img"
            ))
        else:
            results = []
            for i, res in enumerate(executor.map(generate_single_composite, tasks)):
                results.append(res)
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(tasks)} images...")

    # --- Summary ---
    success_count = sum(1 for r in results if r.get("success"))
    fail_count = len(results) - success_count
    total_objects = sum(r.get("num_objects", 0) for r in results if r.get("success"))
    print(f"\nGeneration summary: {success_count} succeeded, {fail_count} failed, {total_objects} total objects placed.")

    # --- Write data.yaml ---
    data_yaml_content = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names
    }
    
    with open(args.output / "data.yaml", "w") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)
    
    print(f"Generation complete. Data saved to {args.output}")
    print(f"Created data.yaml with {len(class_names)} classes.")


if __name__ == "__main__":
    main()