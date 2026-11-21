# Teammate experiments

Three standalone scripts from a teammate's exploratory work. They are **separate from the shipped
model** (`predict.py`, `src/`, `steps/`) and share no code with it. Changes made when adding them to
the repo: default paths now point at `data/`, outputs go to `experiments/results/`, and one dead,
commented-out copy of `run_fft.py` was removed. The analysis logic is unchanged.

Run from the repo root with the project venv (they need `opencv-python`, which is in `requirements.txt`):

```bash
.venv/bin/python experiments/shutter_analysis.py                           # ~1 s
.venv/bin/python experiments/final_cal.py                                  # ~6 s (decodes 30 DNGs)
.venv/bin/python experiments/run_fft.py                                    # default: APC_0216.dng (800 NTU)
.venv/bin/python experiments/run_fft.py data/calibration_dng/APC_0195.dng  # any DNG/JPG/PNG
.venv/bin/python experiments/run_fft.py path/to/img.dng --show             # also open a window
```

Each script takes `--help`, and `--data-dir` / `--out` (or `--out-dir`) to change the paths.

---

## `shutter_analysis.py`: choosing the exposure

- **Data:** `data/shutter_sweep_jpg/`, burst 0 at each of 8 shutter speeds (5 JPGs each, see `data/README.md`).
- **Output:** `results/shutter_speed_analysis.png` (the teammate's original WhatsApp copy is kept next to it as
  `shutter_speed_analysis_teammate_original.jpeg`. The re-run reproduces it.)

| Shutter | Sharpness (Laplacian var) | Contrast (std) | Brightness (0–255) | Clipped px (≥253) |
|---|---|---|---|---|
| 1/200 | 620 | 43.6 | 227.7 | 54.9 % |
| 1/400 | 850 | 63.6 | 196.3 | 30.0 % |
| 1/640 | 1331 | 68.0 | 187.4 | 25.9 % |
| 1/800 | 1568 | 63.6 | 185.1 | 21.5 % |
| 1/1600 | 1617 | **77.2** | 145.4 | 14.0 % |
| 1/3200 | **1676** | 74.3 | 123.6 | 8.1 % |
| 1/5000 | 1429 | 72.9 | 98.8 | 3.9 % |
| 1/8000 | 1116 | 65.6 | 73.7 | **0.44 %** |

**Takeaway:** sharpness and contrast peak at 1/1600–1/3200, but only 1/8000 gets clipping under 0.5 %. This
supports point 5 of the calibration protocol in the main README: shorten the shutter toward
1/4000–1/8000 so the spot core is unclipped. (The October calibration set was shot at 1/2000 s and is
heavily clipped in R.)

Caveats: these are processed JPGs (tone curve, sharpening, 8-bit), not linear RAW, and the NTU of the
sample was not recorded.

## `final_cal.py`: whole-image sharpness baseline

- **Data:** `data/calibration_dng/` (the same 30 DNGs as the main model; its NTU mapping matches `labels.csv`).
- **Output:** `results/cropped_calibration_curves.png` and three quadratic equations printed to the console.

For each DNG it demosaics to white-balanced, gamma-encoded RGB (`rawpy.postprocess`, half size), converts
to gray, keeps the central 50 %, and computes Laplacian variance, the FFT high-frequency energy ratio,
and the standard deviation. It averages these per NTU level and fits `NTU = a·x² + b·x + c` through the 6 points.

**Result:** none of the three metrics changes monotonically with NTU (all three jump at 0–50 NTU and
reverse at 200 NTU), so the fitted quadratics are not usable calibrations. A quadratic through 6
points also leaves only 3 residual degrees of freedom. This is useful as a negative baseline. It
shows why the main pipeline works on the linear blue RAW plane and uses a radial decay length
instead of whole-image sharpness or contrast.

## `run_fft.py`: FFT visualiser

Shows one image, its 2D log-magnitude FFT, and the 1D radial average of the spectrum. It is an
exploratory tool and is not used for any model. Example outputs for 800 NTU (`APC_0216`) and 0 NTU
(`APC_0195`) are in `results/`. Step 5 of the main pipeline (`steps/step5_fft_features.py`) runs the
systematic version of this analysis on the linear B plane.
