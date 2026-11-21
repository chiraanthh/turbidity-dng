"""Shutter-speed sweep: which exposure keeps the laser spot sharp without clipping (teammate's experiment).

For each shutter speed, loads the first burst of 5 JPGs from data/shutter_sweep_jpg/ and averages
brightness, % clipped pixels (>= 253), sharpness (Laplacian variance) and contrast (std).
See data/shutter_sweep_labels.csv for the file -> shutter mapping.

    python experiments/shutter_analysis.py [--data-dir data/shutter_sweep_jpg] [--out experiments/results/...]
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "shutter_sweep_jpg"
OUT = REPO / "experiments" / "results" / "shutter_speed_analysis.png"

def generate_file_lists():
    """Generates the dictionary of filenames based on the provided ranges."""
    # Labels are the Lightroom settings. EXIF records 1/5000 as 1/5300 s and 1/8000 as 1/8900 s.
    ranges = {
        '1/200': (44, 48),
        '1/400': (59, 63),
        '1/640': (74, 78),
        '1/800': (89, 93),
        '1/1600': (109, 113),
        '1/3200': (139, 143),
        '1/5000': (155, 159),
        '1/8000': (170, 174)
    }
    
    speed_data = {}
    for speed, (start, end) in ranges.items():
        # Creates names like 'APC_0044.jpg' by padding with zeros to 4 digits
        speed_data[speed] = [os.path.join(DATA_DIR, f"APC_{i:04d}.jpg") for i in range(start, end + 1)]
        
    return speed_data

def analyze_and_plot_shutter_speeds(speed_data):
    averaged_results = {}
    
    # Data storage for plotting
    speeds_list = []
    avg_sharpness_list = []
    avg_contrast_list = []
    avg_brightness_list = []
    avg_clipped_list = []

    for speed, files in speed_data.items():
        print(f"\nAnalyzing Shutter Speed: {speed}")
        
        metrics = {'brightness': [], 'clipped_pct': [], 'sharpness': [], 'contrast': []}
        
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"  [!] Warning: Could not load '{file}'. Check if it is in the same folder.")
                continue
            
            # Calculate metrics
            metrics['brightness'].append(np.mean(img))
            # Using >= 253 to account for jpg compression artifacts near pure white
            metrics['clipped_pct'].append(np.sum(img >= 253) / img.size * 100)
            metrics['sharpness'].append(cv2.Laplacian(img, cv2.CV_64F).var())
            metrics['contrast'].append(np.std(img))
            
        if metrics['brightness']: 
            avg_brightness = np.mean(metrics['brightness'])
            avg_clipped = np.mean(metrics['clipped_pct'])
            avg_sharpness = np.mean(metrics['sharpness'])
            avg_contrast = np.mean(metrics['contrast'])
            
            averaged_results[speed] = {
                'avg_brightness': avg_brightness,
                'avg_clipped_pct': avg_clipped,
                'avg_sharpness': avg_sharpness,
                'avg_contrast': avg_contrast
            }
            
            speeds_list.append(speed)
            avg_brightness_list.append(avg_brightness)
            avg_clipped_list.append(avg_clipped)
            avg_sharpness_list.append(avg_sharpness)
            avg_contrast_list.append(avg_contrast)
            
            print(f"  -> Avg Brightness: {avg_brightness:.2f}/255")
            print(f"  -> Avg Clipping:   {avg_clipped:.4f}%")
            print(f"  -> Avg Sharpness:  {avg_sharpness:.2f}")
            print(f"  -> Avg Contrast:   {avg_contrast:.2f}")

    # Generate the plots
    if speeds_list:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Sharpness
        plt.subplot(2, 2, 1)
        plt.plot(speeds_list, avg_sharpness_list, marker='o', color='skyblue', linewidth=2)
        plt.title('Average Sharpness (Higher is Better)')
        plt.ylabel('Variance of Laplacian')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Contrast
        plt.subplot(2, 2, 2)
        plt.plot(speeds_list, avg_contrast_list, marker='o', color='lightgreen', linewidth=2)
        plt.title('Average Contrast (Higher is Better)')
        plt.ylabel('Standard Deviation')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Brightness
        plt.subplot(2, 2, 3)
        plt.plot(speeds_list, avg_brightness_list, marker='o', color='coral', linewidth=2)
        plt.title('Average Brightness')
        plt.ylabel('Mean Pixel Value (0-255)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Clipping
        plt.subplot(2, 2, 4)
        plt.plot(speeds_list, avg_clipped_list, marker='o', color='salmon', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='-', label='Max Recommended Clipping (0.5%)')
        plt.title('Clipping / Saturation (Keep < 0.5%)')
        plt.ylabel('% of Pixels Maxed Out')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUT)
        print(f"\n[SUCCESS] Plot saved as '{OUT}'.")
        # plt.show() # Uncomment if you want the window to pop up automatically

    return averaged_results

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    DATA_DIR, OUT = args.data_dir, args.out

    # 1. Automatically generate the list of files based on your ranges
    shutter_speed_data = generate_file_lists()
    
    # 2. Run the analysis and generate the plot
    results = analyze_and_plot_shutter_speeds(shutter_speed_data)