"""2D FFT + 1D radial frequency profile of a single image (teammate's exploratory tool).

Loads a DNG/RAW (via rawpy) or a normal image (via OpenCV) as grayscale, computes the
log-magnitude FFT spectrum and its radial average, and saves a 3-panel figure.

    python experiments/run_fft.py [IMAGE] [--out-dir experiments/results] [--show]

IMAGE defaults to data/calibration_dng/APC_0216.dng (800 NTU).
"""
import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy

REPO = Path(__file__).resolve().parents[1]
IMAGE_NAME = REPO / "data" / "calibration_dng" / "APC_0216.dng"


def compute_radial_profile(magnitude_spectrum):
    """Calculates the 1D radial average profile from the 2D FFT spectrum."""
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(int)

    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr


def load_image_as_grayscale(image_path):
    """Loads standard images via OpenCV or RAW/DNG files via rawpy."""
    ext = os.path.splitext(image_path)[1].lower()

    if ext in ['.dng', '.raw', '.nef', '.cr2', '.arw']:
        with rawpy.imread(str(image_path)) as raw:
            # Postprocess raw data to an RGB numpy array
            rgb = raw.postprocess()
            # Convert RGB to Grayscale for FFT analysis
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        # Standard formats (PNG, JPG, BMP, TIF)
        return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)


def run_fft_analysis(image_path, out_dir, show=False):
    image_path = Path(image_path)
    image_filename = image_path.name

    if not image_path.exists():
        raise FileNotFoundError(f"Could not find '{image_filename}' at path: {image_path}")

    # Load image supporting both regular formats and DNG/RAW formats
    img = load_image_as_grayscale(image_path)

    if img is None:
        raise ValueError(f"Failed to decode image file '{image_filename}'.")

    h, w = img.shape
    print(f"Loaded '{image_filename}' ({w}x{h} - {w*h:,} total pixels scanned).")

    # 1. Compute 2D FFT across ALL pixels
    dft = np.fft.fft2(img)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shifted) + 1)

    # 2. Compute 1D Radial Profile
    radial_profile = compute_radial_profile(magnitude_spectrum)

    # 3. Create Visualization Layout
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Row 1, Left: Original Image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, cmap="gray")
    ax0.set_title(f"Original Image: {image_filename}")
    ax0.axis("off")

    # Row 1, Right: 2D FFT Spectrum
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(magnitude_spectrum, cmap="gray")
    ax1.set_title("2D FFT Spectrum")
    ax1.axis("off")

    # Row 2, Full Width: 1D Radial Profile Plot
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(radial_profile, color="blue", linewidth=1.5)
    ax2.set_title("1D Radial Frequency Profile")
    ax2.set_xlabel("Spatial Frequency (Distance from Center / Radii)")
    ax2.set_ylabel("Log Magnitude")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    # Save resulting plot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"fft_result_{image_path.stem}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Result saved as '{output_path}'")

    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", type=Path, default=IMAGE_NAME)
    ap.add_argument("--out-dir", type=Path, default=REPO / "experiments" / "results")
    ap.add_argument("--show", action="store_true", help="also open the figure in a window")
    a = ap.parse_args()
    if not a.show:
        plt.switch_backend("Agg")
    run_fft_analysis(a.image, a.out_dir, a.show)
