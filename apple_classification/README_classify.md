# Apple Detection, Counting & Color Classification — YOLO11 + HSV

Penghitungan dan klasifikasi warna apel (merah / kuning / hijau) menggunakan **YOLO11**
(COCO pre-trained) + **HSV colour analysis**.

---

## Struktur Project

```
├── classify.py              # ★ Deteksi, hitung & klasifikasi warna apel
├── ai_assignment_20241202_count.jpeg   # Gambar uji penghitungan
├── classify.jpg                        # Gambar uji klasifikasi warna
├── output/                             # Hasil crop + anotasi (auto-generated)
│   ├── red_1.jpg, red_2.jpg, ...       # Crop apel merah
│   ├── yellow_1.jpg, yellow_2.jpg, ... # Crop apel kuning
│   ├── green_1.jpg, green_2.jpg, ...   # Crop apel hijau
│   └── classify_annotated.jpg          # Visualisasi bounding box berwarna
├── requirements_classify.txt
└── README_count.md
```

---

## Quick Start

### Install dependencies
```bash
pip install torch>=2.0.0 ultralytics>=8.2.0 opencv-python>=4.9.0 numpy>=1.26.0
```

### Jalankan deteksi + klasifikasi
```bash
python classify.py
```

### Output
- `output/{red|yellow|green}_{N}.jpg` — Crop setiap apel sesuai warna
- `output/classify_annotated.jpg` — Gambar dengan bounding box berwarna
- Terminal summary: jumlah apel per warna

---

## Pipeline

1. **Deteksi apel** — YOLO11 Large (`yolo11l.pt`) di COCO class 47 (`apple`) dengan TTA (test-time augmentation) dan `imgsz=1280`.
2. **Filter area** — Deteksi dengan luas < `min_crop_area` (200 px²) ditolak.
3. **Klasifikasi warna** — Setiap crop diklasifikasi melalui dua tahap:
   - **HSV majority-pixel voting** — Piksel dicocokkan dengan tiga mask warna (red/yellow/green). Jika ≥4% piksel terklasifikasi, warna tersebut digunakan.
   - **BGR fallback** — Jika voting HSV tidak konklusif, analisis rasio channel BGR digunakan untuk menangani pencahayaan sulit.
4. **Simpan crop** — Setiap apel disimpan sebagai `{warna}_{indeks}.jpg`.
5. **Anotasi visual** — Bounding box berwarna (merah/kuning/hijau) + label indeks.

---

## Parameter

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `input_image` | `classify.jpg` | Gambar input |
| `output_dir` | `output_classify` | Folder hasil |
| `model_name` | `yolo11l.pt` | Model YOLO11 pre-trained COCO |
| `conf_threshold` | 0.08 | Minimum confidence deteksi |
| `iou_threshold` | 0.35 | IoU threshold NMS |
| `imgsz` | 1280 | Resolusi input (deteksi objek kecil) |
| `augment` | `True` | Test-time augmentation |
| `min_crop_area` | 200 | Minimum luas crop (px²) |

### HSV Color Ranges

| Warna | Hue Range | Sat Min | Val Min |
|-------|-----------|---------|---------|
| Red   | 0–12 & 160–180 | 15 | 15 |
| Yellow | 20–28 | 15 | 15 |
| Green | 30–98 | 8 | 8 |

---

## Model Reference

| Model | Source | Keterangan |
|-------|--------|------------|
| **YOLO11 Large** (`yolo11l.pt`) | [Ultralytics](https://github.com/ultralytics/ultralytics) | Pre-trained COCO, class 47 = `apple` |

Model otomatis diunduh (~85 MB) saat first run jika belum tersedia di lokal.

---

## Data Reference

| File | Deskripsi |
|------|-----------|
| `classify.jpg` | Gambar uji klasifikasi & penghitungan apel |
| `ai_assignment_20241202_count.jpeg` | Gambar uji penghitungan apel |

---

## Requirements

- Python 3.12+
- torch ≥ 2.0
- ultralytics ≥ 8.2
- opencv-python ≥ 4.9
- numpy ≥ 1.26

Lihat `requirements_classify.txt` untuk daftar lengkap.

---

## Catatan

- YOLO11 mendeteksi apel sebagai COCO class 47; confidence threshold rendah (0.08) digunakan agar apel hijau yang kurang kontras tetap tertangkap.
- Klasifikasi warna menggabungkan HSV voting (robust untuk warna solid) dan BGR fallback (robust terhadap variasi pencahayaan).
- Area crop minimum (200 px²) menghilangkan false-positive kecil dari latar belakang.
