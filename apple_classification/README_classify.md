# SawitPro AI — Apple Detection & Color Classification

Deteksi dan klasifikasi apel berdasarkan warna (merah / kuning / hijau) menggunakan **YOLO11** (COCO pre-trained) + **HSV colour analysis**.

---

## Requirements

- **OS:** Ubuntu / macOS (UNIX/Linux)
- **Python:** 3.11
- **GPU:** NVIDIA dengan CUDA (opsional; otomatis fallback ke CPU)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_classify.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.0 | PyTorch backend |
| `torchvision` | ≥0.15 | Image transforms |
| `ultralytics` | ≥8.2 | YOLO11 model |
| `opencv-python` | ≥4.9 | Image I/O & drawing |
| `numpy` | ≥1.26 | Numerical operations |
| `pillow` | ≥10.0 | Image loading |
| `transformers` | ≥4.36 | HuggingFace Transformers |

> **Catatan first run:** Pada eksekusi pertama, program akan otomatis mengunduh model weights dari internet (~85 MB untuk YOLO11). Pastikan koneksi internet tersedia.

---

## Usage

Letakkan gambar input di root project:

- `classify.jpg` — gambar berisi apel dengan berbagai warna

```bash
python classify.py
```

### Output

| File | Description |
|---|---|
| `output_classify/red_1.jpg`, `output_classify/red_2.jpg`, ... | Crop apel merah |
| `output_classify/yellow_1.jpg`, `output_classify/yellow_2.jpg`, ... | Crop apel kuning |
| `output_classify/green_1.jpg`, `output_classify/green_2.jpg`, ... | Crop apel hijau |
| `output_classify/classify_annotated.jpg` | Visualisasi debug — bounding box berwarna sesuai kelas |

---

## Configuration

Parameter dapat disesuaikan melalui dictionary `CONFIG` di bagian atas `classify.py`:

| Parameter | Default | Description |
|---|---|---|
| `input_image` | `"classify.jpg"` | Nama file gambar input |
| `output_dir` | `"output_classify"` | Direktori output |
| `model_name` | `"yolo11l.pt"` | Model YOLO11 pre-trained di COCO (class 47 = apple) |
| `conf_threshold` | 0.08 | Minimum confidence untuk deteksi apel |
| `iou_threshold` | 0.35 | IoU threshold untuk NMS |
| `imgsz` | 1280 | Resolusi input (px) untuk deteksi objek kecil |
| `augment` | `True` | Test-time augmentation (flip + merge) |
| `min_crop_area` | 200 | Minimum luas crop (px²) — crop di bawah ini ditolak |

### HSV Color Ranges

Rentang warna HSV dapat disesuaikan jika kondisi pencahayaan berbeda:

| Color | Hue Range | Sat Min | Val Min |
|---|---|---|---|
| Red | 0–12 & 160–180 | 15 | 15 |
| Yellow | 20–28 | 15 | 15 |
| Green | 30–98 | 8 | 8 |

---

## Methodology

### YOLO11 + HSV Classification

COCO (Common Objects in Context) mencakup "apple" sebagai class 47 di antara 80 kelas yang tersedia. YOLO11 yang di-pre-train di COCO dapat mendeteksi apel secara andal di gambar natural.

1. Deteksi semua apel dengan **YOLO11 Large** (`yolo11l.pt`, COCO pre-trained) menggunakan TTA (test-time augmentation).
2. Untuk setiap region yang terdeteksi, crop dan konversi ke ruang warna **HSV**.
3. Klasifikasi melalui **majority-pixel voting** dengan tiga mask warna:
   - **Merah:** Hue 0–12 & 160–180
   - **Kuning:** Hue 20–28
   - **Hijau:** Hue 30–98
4. Jika voting HSV tidak konklusif (<4% piksel terklasifikasi), lakukan **fallback BGR analysis** berbasis rasio channel BGR untuk menangani pencahayaan sulit.
5. Simpan setiap crop dengan label warna yang sesuai + indeks berurutan.
6. Hasilkan visualisasi anotasi dengan bounding box berwarna (merah/kuning/hijau).

---

## Attribution

- **YOLO11** — Ultralytics. [GitHub](https://github.com/ultralytics/ultralytics). Trained on [COCO dataset](https://cocodataset.org).
- **OpenCV** — [opencv.org](https://opencv.org/). Used for image I/O, drawing, and color-space conversion.
- **HuggingFace Transformers** — [huggingface.co](https://huggingface.co/docs/transformers).
