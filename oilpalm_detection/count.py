import argparse
import os
import time
import glob
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_DIR = "Model"
OUTPUT_DIR = "Output/detection_results"
TILE_SIZE = 640
STRIDE = 512
OVERLAP = TILE_SIZE - STRIDE
EDGE_MARGIN = OVERLAP
CONF_THRESHOLD = 0.40
DRAW_CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5
CONTAINMENT_RATIO = 0.85


def tile_starts(length, tile_size, stride):
    """Start positions so every tile is full-size; the last tile is shifted to
    end exactly at the image edge (no undersized edge tiles)."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_image(image, tile_size, stride):
    tiles = []
    img_h, img_w = image.shape[:2]
    for y in tile_starts(img_h, tile_size, stride):
        for x in tile_starts(img_w, tile_size, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, x, y))
    return tiles


def nms(boxes, confs, iou_threshold, containment_ratio=CONTAINMENT_RATIO):
    if len(boxes) == 0:
        return np.array([]), np.array([])
    boxes = np.array(boxes)
    confs = np.array(confs)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        remaining_areas = areas[order[1:]]
        contained_in_kept = inter / (remaining_areas + 1e-8)
        suppress = (iou >= iou_threshold) | (contained_in_kept >= containment_ratio)
        order = order[1:][~suppress]
    return boxes[keep], confs[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect oil palm trees using tiled YOLO inference")
    parser.add_argument("image", nargs="?", default="ai_assignment_20241202_count.jpeg",
                        help="Path to input image")
    args = parser.parse_args()
    image_path = args.image

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
    print(f"Found {len(model_files)} model(s):")
    for m in model_files:
        print(f"  - {os.path.basename(m)}")
    print()

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_h, img_w = image_bgr.shape[:2]
    print(f"Image loaded: {image_path} ({img_w}x{img_h})")
    print(f"Tile size: {TILE_SIZE}, Conf threshold: {CONF_THRESHOLD}, IoU threshold: {IOU_THRESHOLD}\n")

    tiles = tile_image(image_bgr, TILE_SIZE, STRIDE)
    print(f"Image split into {len(tiles)} tiles\n")

    results_summary = []

    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        t_start = time.time()
        model = YOLO(model_path)

        all_boxes = []
        all_confs = []

        for idx, (tile, x_off, y_off) in enumerate(tiles):
            tile_h, tile_w = tile.shape[:2]
            results = model.predict(
                source=tile,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
                imgsz=TILE_SIZE,
            )

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                offset = np.array([x_off, y_off, x_off, y_off])
                tile_right = x_off + tile_w
                tile_bottom = y_off + tile_h
                at_right_edge = (tile_right >= img_w)
                at_bottom_edge = (tile_bottom >= img_h)
                for box in boxes:
                    bx = box.xyxy[0].cpu().numpy() + offset
                    cx = (bx[0] + bx[2]) / 2
                    cy = (bx[1] + bx[3]) / 2
                    if (not at_right_edge and cx > tile_right - EDGE_MARGIN):
                        continue
                    if (not at_bottom_edge and cy > tile_bottom - EDGE_MARGIN):
                        continue
                    all_boxes.append(bx)
                    all_confs.append(float(box.conf[0]))

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(tiles)} tiles...")

        t_elapsed = time.time() - t_start

        raw_count = len(all_boxes)
        filtered_boxes, filtered_confs = nms(all_boxes, all_confs, NMS_IOU_THRESHOLD)
        nms_count = len(filtered_boxes)
        nms_avg_conf = float(np.mean(filtered_confs)) if len(filtered_confs) > 0 else 0.0

        high_conf_mask = filtered_confs >= DRAW_CONF_THRESHOLD
        high_conf_boxes = filtered_boxes[high_conf_mask]
        high_conf_confs = filtered_confs[high_conf_mask]
        num_final = len(high_conf_boxes)
        final_avg_conf = float(np.mean(high_conf_confs)) if len(high_conf_confs) > 0 else 0.0

        print(f"  Raw detections   : {raw_count}")
        print(f"  After NMS dedup  : {nms_count} trees (avg conf: {nms_avg_conf:.4f})")
        print(f"  Conf >= {DRAW_CONF_THRESHOLD:.2f}    : {num_final} trees (avg conf: {final_avg_conf:.4f})")
        print(f"  Time             : {t_elapsed:.2f}s")

        if num_final > 0:
            annotated = image_bgr.copy()
            for i, box in enumerate(high_conf_boxes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                number_label = f"#{i + 1}"
                (nw, nh), _ = cv2.getTextSize(number_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                cv2.rectangle(annotated, (x1, y1 - nh - 6), (x1 + nw + 6, y1), (0, 0, 255), -1)
                cv2.putText(annotated, number_label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                conf_label = f"{high_conf_confs[i]:.2f}"
                (tw, th), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(annotated, (x2 - tw - 4, y1), (x2, y1 + th + 4), (0, 0, 255), -1)
                cv2.putText(annotated, conf_label, (x2 - tw - 2, y1 + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(model_name)[0]}_tiled_result.jpg")
            cv2.imwrite(out_path, annotated)
            print(f"  Saved           : {out_path}")

        results_summary.append({
            "model": model_name,
            "raw": raw_count,
            "nms": nms_count,
            "final": num_final,
            "time": t_elapsed,
            "avg_conf": final_avg_conf,
        })
        print()

    print(f"\n{'='*72}")
    print(f"{'SUMMARY':^72}")
    print(f"{'='*72}")
    print(f"{'Model':<48} {'Raw':>5} {'NMS':>5} {'>={}'.format(DRAW_CONF_THRESHOLD):>6} {'Avg Conf':>9}  {'Time':>8}")
    print("-" * 72)
    for r in results_summary:
        name = r['model'][:45]
        print(f"{name:<48} {r['raw']:>5} {r['nms']:>5} {r['final']:>6} {r['avg_conf']:>9.4f}  {r['time']:>6.2f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
