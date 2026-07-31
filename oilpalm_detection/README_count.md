# Oil Palm Tree Detection — YOLOv12-N & YOLOv26-N

Deteksi pohon kelapa sawit pada citra udara/drone menggunakan **YOLOv12-N** dan **YOLOv26-N**
dengan tiled inference (sliding window).

---

## Struktur Project

```
├── 01_data_preparation.py   # Raster → uint8 + tiling (butuh .tif input)
├── 02_model_train_eval.py   # Training YOLO + evaluasi (butuh Roboflow API)
├── count.py                 # ★ Deteksi + penghitungan pohon (tiled inference)
├── run_all_models.py        # Tiled inference batch (alternatif)
│
├── Model/                   # Model weights (.pt)
│   ├── oilpalm_detection_model_yolov12n.pt
│   └── oilpalm_detection_model_yolov26n.pt
│
├── Oilpalm-Tree-Detection-2/  # Dataset (Roboflow export, YOLO format)
├── images/                    # Gambar untuk dokumentasi
├── Output/detection_results/  # Hasil deteksi (auto-generated)
├── requirements_count.txt
└── README_count.md
```

---

## Quick Start — Deteksi Gambar

### Install dependencies
```bash
pip install torch>=2.0.0 ultralytics>=8.3.0 opencv-python>=4.9.0 numpy>=1.26.0
```

### Jalankan deteksi
```bash
# Deteksi ai_assignment_20241202_count.jpeg dengan semua model di Model/
python count.py
```

### Output
- `Output/detection_results/<model>_tiled_result.jpg` — Gambar dengan bounding box + confidence label

---

## Parameter

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `TILE_SIZE` | 640 | Ukuran tile (px) |
| `STRIDE` | 512 | Langkah sliding window (overlap = 128) |
| `CONF_THRESHOLD` | 0.40 | Confidence threshold YOLO per tile |
| `DRAW_CONF_THRESHOLD` | 0.40 | Confidence threshold final (untuk anotasi) |
| `IOU_THRESHOLD` | 0.40 | IoU threshold untuk NMS internal Ultralytics |
| `NMS_IOU_THRESHOLD` | 0.50 | IoU threshold untuk deduplikasi antar-tile |
| `CONTAINMENT_RATIO` | 0.85 | Rasio containment untuk deduplikasi box nested |

Edit langsung di `count.py` untuk mengganti parameter.

---

## Hasil Deteksi

| Model | Raw | NMS | Conf ≥ 0.40 | Avg Confidence | Time |
|-------|-----|-----|-------------|----------------|------|
| **YOLOv12-N** | **943** | **799** | **799** | 0.6925 | 53.23s |
| **YOLOv26-N** | 217 | 205 | **205** | 0.5760 | 25.75s |

Gambar uji: `ai_assignment_20241202_count.jpeg` (5954×6978 px, 168 tiles 640×640)

> Final count = Conf ≥ 0.40 (setelah NMS deduplication). YOLOv12-N mendeteksi ~3.9× lebih banyak pohon dibanding YOLOv26-N.

---

## Pipeline Inference

1. **Tiling** — Gambar dipotong 640×640 dengan stride 512 (overlap 128 px). Tile terakhir digeser ke tepi gambar (tidak ada tile kecil).
2. **Per-tile YOLO** — Setiap tile di-inference dengan `imgsz=640`.
3. **Koordinat global** — Box lokal dikonversi ke koordinat global; box di area padding diabaikan.
4. **Hybrid deduplication** — Deteksi dari tile berbeda yang overlap digabung menggunakan kombinasi center distance + IoU.
5. **Confidence filter** — Hanya deteksi ≥ `FINAL_CONF` yang disimpan.

---

## Requirements

- Python 3.10+
- torch ≥ 2.0
- ultralytics ≥ 8.3
- opencv-python ≥ 4.9
- numpy ≥ 1.26

Lihat `requirements_count.txt` untuk daftar lengkap.

---

## Catatan

- Model **YOLOv12-N** menghasilkan deteksi ~3× lebih banyak dibanding YOLOv26-N pada dataset yang sama.
- Untuk hasil optimal, gunakan `DRAW_CONF_THRESHOLD=0.40`.
- `01_data_preparation.py` dan `02_model_train_eval.py` membutuhkan data raster / Roboflow API dan hanya digunakan untuk training, bukan inference.

## Reference
- Dataset: [Oilpalm Tree Detection v2](https://universe.roboflow.com/mwahyur/oilpalm-tree-detection-vllsd) — 115 gambar anotasi YOLOv12 format, Roboflow Universe, License: MIT