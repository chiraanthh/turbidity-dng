"""Baseline calibration from whole-image sharpness metrics (teammate's experiment).

Computes three metrics on the central 50 % crop of each calibration DNG (gamma-encoded,
white-balanced RGB -> grayscale): Laplacian variance, FFT high-frequency energy ratio, and
contrast (std). Averages them per NTU level and fits a quadratic NTU = f(metric) to the 6 points.

This is NOT the shipped model (see predict.py / src/). It is kept as a comparison baseline.

    python experiments/final_cal.py [--data-dir data/calibration_dng] [--out experiments/results/...]
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rawpy

REPO = Path(__file__).resolve().parents[1]

def load_dng_grayscale(path):
    """Loads a .dng raw file and converts it to a grayscale numpy array."""
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"  [!] Error loading {path}: {e}")
        return None

def crop_center(image, keep_fraction=0.5):
    """Keeps only the central region of the image, discarding outer edges."""
    h, w = image.shape
    crop_h = int(h * (1.0 - keep_fraction) / 2)
    crop_w = int(w * (1.0 - keep_fraction) / 2)
    return image[crop_h : h - crop_h, crop_w : w - crop_w]

def compute_spatial_blur(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def compute_fft_blur(img, threshold_ratio=0.15):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.ones((rows, cols), np.uint8)
    r = int(min(rows, cols) * threshold_ratio)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
    mask[mask_area] = 0
    
    magnitude_spectrum = np.abs(fshift)
    total_energy = np.sum(magnitude_spectrum)
    if total_energy == 0: return 0.0
    return (np.sum(magnitude_spectrum * mask) / total_energy) * 100

def compute_contrast(img):
    return np.std(img)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "calibration_dng")
    ap.add_argument("--out", type=Path, default=REPO / "experiments" / "results" / "cropped_calibration_curves.png")
    args = ap.parse_args()

    # Your exact file mapping (same as labels.csv at the repo root)
    ntu_batches = {
        400: ['APC_0190.dng', 'APC_0191.dng', 'APC_0192.dng', 'APC_0193.dng', 'APC_0194.dng'],
        0:   ['APC_0195.dng', 'APC_0196.dng', 'APC_0197.dng', 'APC_0198.dng', 'APC_0199.dng'],
        10:  ['APC_0200.dng', 'APC_0201.dng', 'APC_0202.dng', 'APC_0203.dng', 'APC_0204.dng'],
        50:  ['APC_0205.dng', 'APC_0206.dng', 'APC_0207.dng', 'APC_0208.dng', 'APC_0209.dng'],
        200: ['APC_0210.dng', 'APC_0211.dng', 'APC_0212.dng', 'APC_0213.dng', 'APC_0214.dng'],
        800: ['APC_0216.dng', 'APC_0217.dng', 'APC_0218.dng', 'APC_0219.dng', 'APC_0220.dng']
    }

    results = {ntu: {'spatial': [], 'fft': [], 'contrast': []} for ntu in ntu_batches}

    print("Extracting metrics from CROPPED DNG files...")
    for ntu, files in ntu_batches.items():
        print(f"Processing {ntu} NTU batch...")
        for f in files:
            f = os.path.join(args.data_dir, f)
            if not os.path.exists(f):
                print(f"  [!] Missing file: {f}")
                continue
                
            img = load_dng_grayscale(f)
            if img is not None:
                # Apply ROI Crop before computing metrics
                roi_img = crop_center(img, keep_fraction=0.5)
                
                results[ntu]['spatial'].append(compute_spatial_blur(roi_img))
                results[ntu]['fft'].append(compute_fft_blur(roi_img))
                results[ntu]['contrast'].append(compute_contrast(roi_img))

    # Average and Sort Data (0 -> 800)
    sorted_ntu = sorted(ntu_batches.keys())
    avg_spatial = [np.mean(results[ntu]['spatial']) for ntu in sorted_ntu]
    avg_fft = [np.mean(results[ntu]['fft']) for ntu in sorted_ntu]
    avg_contrast = [np.mean(results[ntu]['contrast']) for ntu in sorted_ntu]

    # Generate NEW Calibration Equations
    print("\n=== NEW CALIBRATION EQUATIONS (CROPPED) ===")
    z_spatial = np.polyfit(avg_spatial, sorted_ntu, 2)
    z_fft = np.polyfit(avg_fft, sorted_ntu, 2)
    z_contrast = np.polyfit(avg_contrast, sorted_ntu, 2)
    
    print(f"Spatial Equation:  NTU = {z_spatial[0]:.4e}*x^2 + {z_spatial[1]:.4e}*x + {z_spatial[2]:.4f}")
    print(f"FFT Equation:      NTU = {z_fft[0]:.4e}*x^2 + {z_fft[1]:.4e}*x + {z_fft[2]:.4f}")
    print(f"Contrast Equation: NTU = {z_contrast[0]:.4e}*x^2 + {z_contrast[1]:.4e}*x + {z_contrast[2]:.4f}")

    # Plot the Curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(sorted_ntu, avg_spatial, marker='o', color='blue', linewidth=2)
    plt.title("Cropped Spatial Blur vs Turbidity")
    plt.xlabel("Turbidity (NTU)")
    plt.ylabel("Variance of Laplacian")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(sorted_ntu, avg_fft, marker='o', color='green', linewidth=2)
    plt.title("Cropped FFT Ratio vs Turbidity")
    plt.xlabel("Turbidity (NTU)")
    plt.ylabel("High Freq Energy (%)")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(sorted_ntu, avg_contrast, marker='o', color='red', linewidth=2)
    plt.title("Cropped Contrast vs Turbidity")
    plt.xlabel("Turbidity (NTU)")
    plt.ylabel("Standard Deviation")
    plt.grid(True)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    print(f"\n[SUCCESS] Saved plots to '{args.out}'.")