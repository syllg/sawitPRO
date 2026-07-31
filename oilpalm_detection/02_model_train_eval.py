"""
02_model_train_eval.py
----------------------
Migrasi dari 02_model_train_eval.ipynb (Colab) ke script lokal.

Fungsi:
  1. Download dataset dari Roboflow (YOLO format).
  2. Training model YOLO pada dataset oil palm.
  3. Validasi model (mAP, precision, recall).
  4. Evaluasi per-image dengan IoU matching.

Usage:
  python 02_model_train_eval.py
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "mwahyur")
ROBOFLOW_PROJECT = os.environ.get("ROBOFLOW_PROJECT", "oilpalm-tree-detection-vllsd")
ROBOFLOW_VERSION = int(os.environ.get("ROBOFLOW_VERSION", "2"))

MODEL_SAVE_PATH = "Model/oilpalm_detection_model_yolov26n.pt"
DATA_YAML = "Oilpalm-Tree-Detection-2/data.yaml"

EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16


def download_dataset() -> str:
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Install roboflow: pip install roboflow")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download("yolo26")
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model_path: str, data_yaml: str, device: str) -> YOLO:
    model = YOLO(model_path)
    print(f"Training on device: {device}")
    model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=device)
    return model


def validate_model(model: YOLO, data_yaml: str) -> None:
    metrics = model.val(data=data_yaml)
    print("=" * 40)
    print("VALIDATION RESULTS")
    print("=" * 40)
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def evaluate_per_image(model: YOLO, valid_folder: str) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt

    valid_images = sorted([
        os.path.join(valid_folder, img)
        for img in os.listdir(valid_folder)
        if img.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])
    print(f"Validation images: {len(valid_images)}")

    total_tp, total_fp, total_fn = 0, 0, 0
    all_true_labels = []
    all_predicted_scores = []

    for idx, img_path in enumerate(valid_images, 1):
        results = model.predict(source=img_path, save=False, verbose=False)

        label_path = img_path.replace("images", "labels").replace(
            os.path.splitext(img_path)[1], ".txt"
        )
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append([float(v) for v in parts])

        predicted_boxes = []
        pred_scores = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                predicted_boxes.append(box.xyxy[0].tolist())
                pred_scores.append(box.conf.item())

        tp, fp = 0, 0
        fn = len(gt_boxes)
        img_h, img_w = results[0].orig_shape

        for pred_box, score in zip(predicted_boxes, pred_scores):
            matched = False
            true_label = 0
            for gt in gt_boxes:
                gt_xmin = int((gt[1] - gt[3] / 2) * img_w)
                gt_ymin = int((gt[2] - gt[4] / 2) * img_h)
                gt_xmax = int((gt[1] + gt[3] / 2) * img_w)
                gt_ymax = int((gt[2] + gt[4] / 2) * img_h)
                iou = calculate_iou(pred_box, [gt_xmin, gt_ymin, gt_xmax, gt_ymax])
                if iou > 0.5:
                    matched = True
                    true_label = 1
                    break
            if matched:
                tp += 1
                fn -= 1
            else:
                fp += 1
            all_true_labels.append(true_label)
            all_predicted_scores.append(score)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if idx <= 3:
            print(f"  {os.path.basename(img_path)}: GT={len(gt_boxes)} Pred={len(predicted_boxes)} TP={tp} FP={fp} FN={fn}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 40)
    print("PER-IMAGE EVALUATION")
    print("=" * 40)
    print(f"Total TP      : {total_tp}")
    print(f"Total FP      : {total_fp}")
    print(f"Total FN      : {total_fn}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")

    if sum(all_true_labels) > 0:
        precision_curve, recall_curve, _ = precision_recall_curve(all_true_labels, all_predicted_scores)
        ap = average_precision_score(all_true_labels, all_predicted_scores)
        print(f"Average Precision: {ap:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs("Output", exist_ok=True)
        plt.savefig("Output/precision_recall_curve.png", dpi=150)
        plt.close()
        print("Saved: Output/precision_recall_curve.png")


def main() -> None:
    print("=" * 50)
    print("02 — MODEL TRAINING & EVALUATION (lokal)")
    print("=" * 50)

    device = get_device()
    print(f"Device: {device}")

    download_dataset()

    model = train_model("Model/yolo26n.pt", DATA_YAML, device)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved: {MODEL_SAVE_PATH}")

    validate_model(model, DATA_YAML)

    valid_folder = os.path.join(os.path.dirname(DATA_YAML), "valid", "images")
    if os.path.isdir(valid_folder):
        evaluate_per_image(model, valid_folder)

    print("\nDone.")


if __name__ == "__main__":
    main()
