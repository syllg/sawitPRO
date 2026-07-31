"""
01_data_preparation.py
-----------------------
Migrasi dari 01_data_preparation.ipynb (Colab) ke script lokal.

Fungsi:
  1. Membaca raster .tif, menampilkan metadata.
  2. Konversi raster float32 -> uint8 dengan brightness/contrast/gamma adjustment.
  3. Tiling raster menjadi 640x640 PNG tiles.

Usage:
  python 01_data_preparation.py
"""

import os
import numpy as np
import cv2

INPUT_TIF = "raster_data/raster_data_3.tif"
OUTPUT_TIF = "raster_data/raster_data_3_uint8.tif"
TILE_SIZE = 640
TILES_OUT = "raster_data/tiles/png"

BRIGHTNESS = 30
CONTRAST = 1.0
GAMMA = 1.2


def show_metadata(path: str) -> None:
    try:
        import rasterio
    except ImportError:
        print("[SKIP] rasterio tidak terinstall. Install: pip install rasterio")
        return

    if not os.path.exists(path):
        print(f"[SKIP] File tidak ditemukan: {path}")
        return

    with rasterio.open(path) as src:
        print("=" * 50)
        print("METADATA RASTER")
        print("=" * 50)
        print(f"CRS         : {src.crs}")
        print(f"Dimensi     : {src.height} x {src.width}")
        print(f"Bands       : {src.count}")
        print(f"Data type   : {src.dtypes}")
        print(f"NoData      : {src.nodata}")
        print(f"Transform   : {src.transform}")
        for i in range(1, src.count + 1):
            band = src.read(i)
            print(f"  Band {i}: min={band.min()}, max={band.max()}, mean={band.mean():.2f}")


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def convert_to_uint8(
    input_path: str,
    output_path: str,
    brightness: float = BRIGHTNESS,
    contrast: float = CONTRAST,
    gamma: float = GAMMA,
) -> None:
    try:
        import rasterio
    except ImportError:
        print("[SKIP] rasterio tidak terinstall.")
        return

    if not os.path.exists(input_path):
        print(f"[SKIP] File tidak ditemukan: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with rasterio.open(input_path) as src:
        print(f"\nProcessing: {os.path.basename(input_path)}")
        print(f"Bands: {src.count}, Size: {src.height}x{src.width}")

        image_data = src.read([1, 2, 3])

        nodata = src.nodata
        if nodata is not None:
            image_data = np.where(image_data == nodata, np.nan, image_data)

        min_val = np.nanpercentile(image_data, 2, axis=(1, 2), keepdims=True)
        max_val = np.nanpercentile(image_data, 98, axis=(1, 2), keepdims=True)
        scale_range = max_val - min_val
        scale_range[scale_range == 0] = 1
        image_normalized = ((image_data - min_val) / scale_range * 255).astype(np.float32)

        image_adjusted = (image_normalized * contrast + brightness).clip(0, 255).astype(np.uint8)

        image_gamma = np.zeros_like(image_adjusted, dtype=np.uint8)
        for i in range(3):
            image_gamma[i] = gamma_correction(image_adjusted[i], gamma)

        image_gamma = np.nan_to_num(image_gamma, nan=0).astype(np.uint8)

        out_meta = src.meta.copy()
        out_meta.update({"dtype": "uint8", "count": 3, "nodata": 0})

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(image_gamma)

    print(f"Output: {output_path}")


def tile_raster_to_png(
    tif_path: str,
    tiles_out: str,
    tile_size: int = TILE_SIZE,
) -> None:
    try:
        from patchify import patchify
        import tifffile as tiff
        from PIL import Image
    except ImportError:
        print("[SKIP] patchify/tifffile/Pillow tidak terinstall.")
        return

    if not os.path.exists(tif_path):
        print(f"[SKIP] File tidak ditemukan: {tif_path}")
        return

    os.makedirs(tiles_out, exist_ok=True)

    large_image_stack = tiff.imread(tif_path)
    tiles_img = patchify(
        large_image_stack,
        (tile_size, tile_size, large_image_stack.shape[2]),
        step=tile_size,
    )

    img_idx = 0
    for i in range(tiles_img.shape[0]):
        for j in range(tiles_img.shape[1]):
            img_patch = tiles_img[i, j, :, :, :].squeeze().astype(np.uint8)
            img_patch_rgb = img_patch[:, :, :3]
            img = Image.fromarray(img_patch_rgb, "RGB")
            img.save(os.path.join(tiles_out, f"{img_idx}.png"))
            img_idx += 1

    print(f"Saved {img_idx} tiles -> {tiles_out}")


def main() -> None:
    print("=" * 50)
    print("01 — DATA PREPARATION (lokal)")
    print("=" * 50)

    show_metadata(INPUT_TIF)
    convert_to_uint8(INPUT_TIF, OUTPUT_TIF)
    tile_raster_to_png(OUTPUT_TIF, TILES_OUT)

    print("\nDone.")


if __name__ == "__main__":
    main()
