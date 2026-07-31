# SawitPRO 

Dua program AI: **palm tree counting** dari citra drone dan **apple color classification** dari gambar buah.

---

## 1. Program Count — Palm Tree Detection

Mendeteksi dan menghitung pohon kelapa sawit pada `ai_assignment_20241202_count.jpeg` menggunakan **YOLOv12-N** & **YOLOv26-N** dengan tiled inference. Output berupa gambar JPG dengan bounding box bernomor.

| Model | Raw | NMS | Conf ≥ 0.40 | Avg Conf | Time |
|-------|-----|-----|-------------|----------|------|
| YOLOv12-N | 943 | 799 | **799** | 0.6925 | 53.23s |
| YOLOv26-N | 217 | 205 | **205** | 0.5760 | 25.75s |

### Run
```bash
cd oilpalm_detection
pip install torch>=2.0.0 ultralytics>=8.3.0 opencv-python>=4.9.0 numpy>=1.26.0
python count.py
```

### Output
- `Output/detection_results/<model>_tiled_result.jpg` — Gambar dengan bounding box bernomor + confidence

---

## 2. Program Classify — Apple Color Classification

Mendeteksi dan mengklasifikasi apel pada `classify.jpg` menjadi **red**, **yellow**, atau **green** menggunakan **YOLO11** (COCO class 47) + HSV analysis. Output berupa crop per apel berdasarkan warna.

### Run
```bash
cd apple_classification
pip install torch>=2.0.0 ultralytics>=8.2.0 opencv-python>=4.9.0 numpy>=1.26.0
python classify.py
```

### Output
- `output/{red|yellow|green}_{N}.jpg` — Crop setiap apel sesuai warna
- `output/classify_annotated.jpg` — Visualisasi bounding box berwarna

---

## System Requirements

| Requirement | Version |
|-------------|---------|
| OS | UNIX/Linux (Ubuntu, macOS) |
| Python | 3.11+ |
| GPU | NVIDIA CUDA (opsional, fallback ke CPU) |

---

## Attribution

### Models
| Model | Source | License |
|-------|--------|---------|
| YOLOv12-N / YOLOv26-N | Custom trained, [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| YOLO11 Large | [Ultralytics](https://github.com/ultralytics/ultralytics), pre-trained on [COCO](https://cocodataset.org) | AGPL-3.0 |

### Datasets
| Dataset | Source | License |
|---------|--------|---------|
| [Oilpalm Tree Detection v2](https://universe.roboflow.com/mwahyur/oilpalm-tree-detection-vllsd) | Roboflow Universe, 115 annotated images | MIT |
| COCO 2017 | [cocodataset.org](https://cocodataset.org) | CC BY 4.0 |

### Libraries
| Library | Purpose |
|---------|---------|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO model training & inference |
| [OpenCV](https://opencv.org) | Image I/O, drawing, color-space conversion |
| [PyTorch](https://pytorch.org) | Deep learning backend |
| [NumPy](https://numpy.org) | Numerical operations |

---

## Project Structure

```
SawitPRO/
├── oilpalm_detection/          # Program Count — Palm tree detection
│   ├── count.py                # Tiled YOLO inference + NMS counting
│   ├── Model/
│   │   ├── oilpalm_detection_model_yolov12n.pt
│   │   └── oilpalm_detection_model_yolov26n.pt
│   ├── ai_assignment_20241202_count.jpeg
│   ├── requirements_count.txt
│   └── README_count.md
│
├── apple_classification/       # Program Classify — Apple color classification
│   ├── classify.py             # YOLO11 detection + HSV color classification
│   ├── classify.jpg
│   ├── ai_assignment_20241202_count.jpeg
│   ├── requirements_classify.txt
│   └── README_count.md
│
└── README.md                        
```

