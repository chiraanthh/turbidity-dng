# Datasets

Two image sets, both shot on an **iPhone 16 main camera (26 mm, f/1.6, ISO 40)** using Lightroom iOS RAW
capture, looking straight down a red laser beam through a cuvette. File names (`APC_xxxx`) are the
original camera names. They are left unchanged because every output table and plot in the repo
refers to them. Labels are in the CSV files described below.

| Folder | Files | Size | Labels | Used by |
|---|---|---|---|---|
| `calibration_dng/` | 30 Bayer-RAW `.dng` | ~440 MB | [`../labels.csv`](../labels.csv) | main pipeline (`steps/`, `predict.py`, app), `experiments/final_cal.py` |
| `shutter_sweep_jpg/` | 120 `.jpg` | ~150 MB | [`shutter_sweep_labels.csv`](shutter_sweep_labels.csv) | `experiments/shutter_analysis.py` |

---

## 1. `calibration_dng/`: turbidity calibration set (13 Oct 2026)

6 turbidity standards × 5 burst shots = 30 images. Each NTU level is **one** cuvette fill (one
independent sample). The 5 shots are a burst of that same fill.

| NTU | sample_id | Files |
|---:|---|---|
| 0 | `S1_0NTU` | `APC_0195` – `APC_0199` |
| 10 | `S2_10NTU` | `APC_0200` – `APC_0204` |
| 50 | `S3_50NTU` | `APC_0205` – `APC_0209` |
| 200 | `S4_200NTU` | `APC_0210` – `APC_0214` |
| 400 | `S0_400NTU` | `APC_0190` – `APC_0194` |
| 800 | `S5_800NTU` | `APC_0216` – `APC_0220` |

- 400 NTU was shot first, which is why it is `S0` and has the lowest file numbers.
- `APC_0215` does not exist (skipped / discarded during capture).
- `APC_0214` (200 NTU) was taken after the setup shifted. It is the one image the final model misses.
  See the main README.

`labels.csv` (repo root) columns: `file, ntu, sample_id, shot`. The pipeline reads it from there.

- **Capture:** shutter 1/2000 s, ISO 40, f/1.6, Lightroom iOS 11.6.1, 13 Oct 2026 15:49–17:02.
- **Format:** Bayer RAW DNG (12-bit, 4032 × 3024, BGGR mosaic). These are not ProRAW, and the pipeline rejects ProRAW.

---

## 2. `shutter_sweep_jpg/`: shutter-speed sweep (22 Sep 2026, teammate)

An earlier exposure study to choose a shutter speed that keeps the laser spot sharp without clipping.
There are 8 shutter speeds × 3 bursts × 5 shots = 120 JPGs. ISO 40 and f/1.6 throughout.

| Shutter (Lightroom setting) | EXIF exposure | Burst 0 (used in analysis) | Burst 1 | Burst 2 |
|---|---|---|---|---|
| 1/200 | 1/200 | `APC_0044`–`0048` | `0049`–`0053` | `0054`–`0058` |
| 1/400 | 1/400 | `APC_0059`–`0063` | `0064`–`0068` | `0069`–`0073` |
| 1/640 | 1/640 | `APC_0074`–`0078` | `0079`–`0083` | `0084`–`0088` |
| 1/800 | 1/800 | `APC_0089`–`0093` | `0094`–`0098` | `0099`–`0103` |
| 1/1600 | 1/1600 | `APC_0109`–`0113` | `0114`–`0118` | `0119`–`0123` |
| 1/3200 | 1/3200 | `APC_0139`–`0143` | `0144`–`0148` | `0149`–`0153` |
| 1/5000 | **1/5300** | `APC_0155`–`0159` | `0160`–`0164` | `0165`–`0169` |
| 1/8000 | **1/8900** | `APC_0170`–`0174` | `0175`–`0179` | `0180`–`0184` |

`shutter_sweep_labels.csv` columns: `file, shutter_label, exif_exposure_s, iso, fnumber, burst,
used_in_shutter_analysis, datetime`. The shutter values come from each file's EXIF.

**Known unknowns:**
- The sample (NTU) in the cuvette for this session was **not recorded**. Bursts 1 and 2 at each speed were
  taken 8–70 s after the previous burst. Whether they are re-seated cuvettes or different samples is undocumented,
  so only burst 0 is used in the analysis.
- The two fastest speeds are labelled by their Lightroom setting (1/5000, 1/8000), but the EXIF records
  1/5300 s and 1/8900 s.
- The numbering has gaps (`APC_0104`–`0108`, `0124`–`0138`, `0154`). Those shots are not in the set.

---

## Privacy note

The DNGs keep their original EXIF metadata, including GPS location tags on 25 of the 30 calibration files.
The JPGs carry no GPS. Strip GPS (for example with
`exiftool -gps:all= -overwrite_original data/calibration_dng/*.dng`) before making this repository public.
