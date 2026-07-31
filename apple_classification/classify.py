"""Apple Detection & Color Classification using YOLO11 + HSV analysis.

Approach:
  1. Detect all apples in the image using YOLO11 pre-trained on COCO
     (COCO class 47 = "apple").
  2. For each detected apple:
     a. Crop the bounding-box region from the original image.
     b. Classify colour via HSV pixel-voting + BGR fallback.
     c. Save the crop with colour label and sequential index.

Output: output/{red|yellow|green}_{N}.jpg
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration — tweak these values as needed
# ---------------------------------------------------------------------------
CONFIG: dict = {
    # I/O
    "input_image": "classify.jpg",
    "output_dir": "output_classify",
    # Model
    "model_name": "yolo11l.pt",  # YOLO11 pre-trained on COCO (class 47 = apple)
    "conf_threshold": 0.08,  # lower — catch more apples including green ones
    "iou_threshold": 0.35,  # lowered — allow closer boxes
    "imgsz": 1280,  # larger input resolution for small-object detection
    "augment": True,  # test-time augmentation (flip + merge)
    # HSV classification boundaries  (OpenCV H=[0..179], S=[0..255], V=[0..255])
    "color_ranges": {
        "red":    {"hue_low": 0,  "hue_high": 12, "sat_min": 15, "val_min": 15},
        "yellow": {"hue_low": 20, "hue_high": 28, "sat_min": 15, "val_min": 15},
        "green":  {"hue_low": 30, "hue_high": 98, "sat_min": 8,  "val_min": 8},
    },
    "red_high_hue_range": (160, 180),
    # Minimum crop area (pixels) — reject detections smaller than this
    "min_crop_area": 200,  # lowered — catch smaller apples
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def classify_color(crop_bgr: np.ndarray) -> str:
    """Classify a cropped apple as 'red', 'yellow', or 'green'.

    Primary signal: mean BGR channel ratios — robust to lighting variation.
    Secondary signal: HSV majority-pixel voting for confirmation.

    Returns "red" | "yellow" | "green".
    """
    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]

    # --- 1. HSV majority voting (catches clear-cut cases) ---
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    masks: dict[str, np.ndarray] = {}
    for color, cfg in CONFIG["color_ranges"].items():
        sat_mask = s >= cfg["sat_min"]
        val_mask = v >= cfg["val_min"]
        if color == "red":
            hue_mask = (h <= cfg["hue_high"]) | (h >= CONFIG["red_high_hue_range"][0])
        else:
            hue_mask = (h >= cfg["hue_low"]) & (h <= cfg["hue_high"])
        masks[color] = sat_mask & val_mask & hue_mask

    counts = {c: int(m.sum()) for c, m in masks.items()}
    best_hsv = max(counts, key=counts.get)
    if counts[best_hsv] / total_pixels >= 0.04:
        return best_hsv

    # --- 2. Fallback: mean BGR analysis ---
    m = crop_bgr.reshape(-1, 3).astype(np.float32).mean(axis=0)
    b, g, r = m

    # Red apple: R-channel clearly dominates
    if r > g * 1.35 and r > b * 1.4:
        return "red"
    # Green apple: G dominates R  (relaxed — apples with green cast)
    if g > r * 1.02 and g > b * 1.05:
        return "green"
    # Yellow apple: R / G balanced, both > B
    if r > b * 1.2 and g > b * 1.2:
        return "yellow"
    # Default ratio match
    rg_ratio = r / max(g, 1)
    gr_ratio = g / max(r, 1)
    if rg_ratio > 1.2:
        return "red"
    if gr_ratio > 1.01:
        return "green"
    return "yellow"


def draw_bbox_visual(
    image: np.ndarray,
    boxes: np.ndarray,
    colors: list[str],
) -> np.ndarray:
    """Draw coloured bounding boxes + labels (for optional debug viz)."""
    out = image.copy()
    color_map = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
    }
    for idx, ((x1, y1, x2, y2), color) in enumerate(zip(boxes, colors)):
        bgr = color_map.get(color, (255, 255, 255))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2)
        cv2.putText(
            out,
            f"{idx + 1}:{color}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            bgr,
            2,
        )
    return out


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main() -> None:
    output_path = Path(CONFIG["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load image ---
    image = load_image(CONFIG["input_image"])
    print(f"[INFO] Image size: {image.shape[1]}x{image.shape[0]}")

    # --- Load YOLOv8 ---
    print(f"[INFO] Loading model: {CONFIG['model_name']} ...")
    model = YOLO(CONFIG["model_name"])

    # --- Detect apples ---
    print("[INFO] Running apple detection ...")
    results = model(
        CONFIG["input_image"],
        conf=CONFIG["conf_threshold"],
        iou=CONFIG["iou_threshold"],
        imgsz=CONFIG["imgsz"],
        augment=CONFIG["augment"],
        verbose=False,
    )

    # Collect apple detections
    apple_boxes: list[tuple[int, int, int, int]] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            if cls_name == "apple":
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                apple_boxes.append((x1, y1, x2, y2))

    if not apple_boxes:
        print("[WARN] No apples detected. Try lowering conf_threshold.")
        return

    print(f"[INFO] Detected {len(apple_boxes)} apple(s)")

    # --- Classify and save each apple ---
    apple_boxes_array = np.array(apple_boxes)
    colors: list[str] = []
    counters: dict[str, int] = {"red": 0, "yellow": 0, "green": 0}

    for i, (x1, y1, x2, y2) in enumerate(apple_boxes):
        w, h = x2 - x1, y2 - y1
        if w * h < CONFIG["min_crop_area"]:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        color = classify_color(crop)
        colors.append(color)
        counters[color] += 1

        filename = f"{color}_{counters[color]}.jpg"
        out_file = output_path / filename
        cv2.imwrite(str(out_file), crop)
        print(f"  [{i + 1:2d}] {filename}  <- saved")

    # --- Optional: save a debug visualisation ---
    annotated = draw_bbox_visual(image, apple_boxes_array, colors)
    cv2.imwrite(str(output_path / "classify_annotated.jpg"), annotated)

    # --- Summary ---
    print(f"\n[DONE] Summary — {len(apple_boxes)} apples classified:")
    for color, n in counters.items():
        print(f"  {color}: {n}")


if __name__ == "__main__":
    main()
