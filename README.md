# Laser-blur turbidity meter (iPhone Bayer RAW)

Estimates water turbidity (NTU) from how far a red laser's light spreads after passing through a
cuvette, photographed straight down the beam with an iPhone 16 main camera as a Bayer-RAW DNG.

> **Status: proof of concept.** It is calibrated on **6 independent samples** (one cuvette each at
> 0, 10, 50, 200, 400, 800 NTU). It quantifies **200–800 NTU only**. Anything below ~156 NTU is
> reported as "below the quantifiable range" without a number: 0–50 NTU look identical, and 50–200 NTU
> was never measured.
> Read [Limitations](#limitations) before trusting a number.

## Quick start

**Requirements:** Python **3.12** (dependencies are pinned in `requirements.txt` to the versions the model
was trained with), git, and about 1.5 GB of free disk space (the repo holds about 600 MB of images, and the
pipeline caches decoded mosaics).

### 1. Get the code and data

```bash
git clone https://github.com/chiraanthh/turbidity-dng.git
cd turbidity-dng
```

The dataset is committed in `data/` (plain git, no LFS), so there is nothing else to download.
See [`data/README.md`](data/README.md) for what each image is.

### 2. Create the environment

macOS / Linux:
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The commands below use the `.venv/bin/...` form. On Windows use `.venv\Scripts\...`, or activate the venv
first (`source .venv/bin/activate` or `.venv\Scripts\Activate.ps1`) and drop the prefix.

### 3. Predict turbidity

```bash
# CLI, text output (one or more DNGs)
.venv/bin/python predict.py data/calibration_dng/APC_0216.dng data/calibration_dng/APC_0210.dng data/calibration_dng/APC_0195.dng

# CLI, JSON output
.venv/bin/python predict.py data/calibration_dng/APC_0216.dng --json

# Your own photo
.venv/bin/python predict.py path/to/IMG_0001.dng
```

Example output:

```
APC_0216.dng: [ok] 839 NTU (1-sd range 348-2022)
APC_0210.dng: [ok] 224 NTU (1-sd range 93-541)
APC_0195.dng: [ok] below the quantifiable range (< ~156 NTU). Calibration only resolves 200-800 NTU; 0-50 NTU look identical and 50-200 NTU was never measured
```

(These are training images, so they show the tool working, not validated accuracy.)

### 4. Web app (upload DNGs in the browser)

```bash
.venv/bin/streamlit run src/app.py
```

Open the URL it prints (usually http://localhost:8501) and upload a DNG, for example
`data/calibration_dng/APC_0216.dng`.

### 5. (Optional) Re-run the whole analysis from scratch

```bash
for s in steps/step*.py; do .venv/bin/python "$s"; done   # steps 1–8, regenerates outputs/ and models/
.venv/bin/python steps/make_report.py                     # regenerates explanation.md + outputs/tables/
```

The first run decodes all 30 DNGs into `outputs/cache/` (~730 MB, git-ignored). Later runs reuse the cache.

### 6. (Optional) Teammate experiments

```bash
.venv/bin/python experiments/shutter_analysis.py   # shutter-speed sweep → experiments/results/shutter_speed_analysis.png
.venv/bin/python experiments/final_cal.py          # whole-image sharpness baseline → experiments/results/cropped_calibration_curves.png
.venv/bin/python experiments/run_fft.py [IMAGE]    # FFT visualiser for one image
```

See [`experiments/README.md`](experiments/README.md) for what each one shows.

Each result has a `status`:
- `ok`: clean.
- `ok_with_warnings`: for example, EXIF differs from calibration, the spot is near the edge, or the feature is outside the calibration range.
- `rejected`: no NTU is reported, because the spot wasn't found, the fit band is clipped, or the ROI is off the frame.
- `error`: the file is unreadable, ProRAW, or has an unsupported CFA.

## How it works

```
DNG ─► B CFA plane (linear, black-subtracted, no demosaic, no WB)
    ─► spot centre (intensity-weighted centroid on the smoothed B plane)
    ─► fixed 1200×1200 ROI (plane px), pixels ≥ white−64 DN masked as saturated
    ─► radial profile (5 px rings; a ring with >1 % saturated pixels is discarded)
    ─► decay_len_inner = −1 / slope of ln(I) vs r over r = 300–450 px
    ─► model: log10(NTU) = a + b·ln(decay_len_inner) fitted on 200–800 NTU,
              below a learned threshold → "below the quantifiable range" (a bound, no number)
```

**Why the B channel.** The image is a large defocused disc. About 25 % of R pixels in the ROI are clipped
at every NTU, so R carries no usable core. Red light leaks into B pixels at roughly 7 % of R, which works as
a built-in ND filter. B is still clipped out to r ≈ 280 px for 0–50 NTU, which is why the fit band
starts at 300 px.

**Why one feature.** Edge sharpness, decay lengths, core/halo ratio, Laplacian variance, Tenengrad and
spectral slope all correlate at |r| ≥ 0.96 across the 6 samples. They all measure the same thing: scattering
spreads light outward and lengthens the decay. `decay_len_inner` had the best worst-case separation
between adjacent levels and a 1 % spread across shots. Peak-based features (FWHM, 10/90 width, inflection,
peak-to-halo) need an unclipped peak, which only 800 NTU has.

**Single code path.** `src/pipeline.py` is used by training, the robustness tests, the CLI and the app.
`steps/step7_model.py` asserts that it reproduces the step-4 features (difference ~1e-14). All
preprocessing numbers are saved to `models/params.json`, and the model refuses to load if the code's
parameters differ.

## Project layout

| Path | What |
|---|---|
| `predict.py` | CLI |
| `src/app.py` | Streamlit app |
| `src/pipeline.py` | DNG → features + QC (the single feature path) |
| `src/service.py` | `analyse()`: features + model + QC verdict |
| `src/model.py` | target transform, floor model, `load_bundle`, `predict_ntu` |
| `src/loaders.py`, `preprocess.py`, `profiles.py`, `features.py` | building blocks |
| `steps/step1…8_*.py` | the analysis, one script per step (run in order) |
| `models/` | `turbidity_model.joblib` + `params.json` |
| `steps/make_report.py` | builds `explanation.md` + `outputs/tables/` from `docs/explanation_template.md` |
| `models/` | `turbidity_model.joblib` + `params.json` |
| `outputs/` | plots and tables per step; `outputs/cache/` intermediates (mosaic `.npy` files are git-ignored) |
| `explanation.md` | full write-up of the project (generated, so edit the template instead) |
| `labels.csv` | calibration labels: file → ntu, sample_id, shot |
| `data/calibration_dng/` | **dataset:** the 30 calibration DNGs (6 NTU levels × 5 shots), see [`data/README.md`](data/README.md) |
| `data/shutter_sweep_jpg/` | **dataset:** 120 JPGs from the shutter-speed sweep (8 speeds × 3 bursts × 5) |
| `data/shutter_sweep_labels.csv` | shutter-sweep labels (from EXIF) |
| `experiments/` | teammate experiments (shutter sweep, sharpness baseline, FFT viewer) + `results/`, see [`experiments/README.md`](experiments/README.md) |

The pipeline reads calibration DNGs from `data/calibration_dng/` by default. Set `TURBIDITY_DATA_DIR`
to use a different folder.

## Teammate experiments (summary)

- **Shutter-speed sweep** (`experiments/shutter_analysis.py`). On 8 shutter speeds from 1/200 to 1/8000,
  sharpness peaks around 1/1600–1/3200, but only 1/8000 keeps clipped pixels under 0.5 %. This is the
  evidence behind point 5 of the calibration protocol below (shorter shutter so the spot core is unclipped).
- **Whole-image sharpness baseline** (`experiments/final_cal.py`). Laplacian variance, FFT high-frequency
  ratio and contrast of the central crop are **not monotonic** in NTU on the calibration set, so the
  quadratic fits it prints are not usable calibrations. It is kept as a negative baseline that supports
  the radial decay-length feature used here.
- **FFT viewer** (`experiments/run_fft.py`). Shows the 2D spectrum and radial profile of any image. It is
  exploratory only.

## Results (honest numbers)

Leave-one-sample-out CV (6 folds; each fold holds out a whole NTU level, so 0 and 800 are pure
extrapolation):

| Model | MAE (NTU) | RMSE (NTU) | R² | RMSE log10(NTU+1) |
|---|---|---|---|---|
| null (training mean) | 248 | 357 | −0.57 | 1.19 |
| log-log, 1 feature, all levels | 783 | 1756 | −37 | 0.86 |
| **log-log ≥200 + floor (final)** | **49** | **67** | **0.95** | **0.79** |
| Ridge, 2 features | 497 | 1059 | −12.8 | 0.83 |
| GBM, 2 features | 147 | 206 | 0.48 | 1.11 |

**Final model, held out:**
- 0/10/50 NTU images: 100 % called "below the quantifiable range".
- 200/400/800 NTU images: 14 of 15 quantified.
- Mean predictions: 200 → 217, 400 → 358, 800 → 882.
- The one miss is APC_0214, the 200 NTU shot taken after the setup shifted. Including it, the error band is ×/÷ 2.4 (1 sd). For correctly classified images, the typical error is ×/÷ 1.23.
- The ×/÷ 2.4 is what the app and CLI report.

Why the alternatives fail:
- **All-level log-log and Ridge** are forced steep by the flat 0–50 cluster and extrapolate 800 NTU to 3000–5000.
- **GBM** cannot extrapolate: trees predict a constant beyond the training range.
- **R² = 0.95 for the final model rests on 3 quantifiable levels.** It is not a validated accuracy.

**Robustness (step 8)** is the worst case over the 200–800 NTU samples, as change in predicted NTU:

| Perturbation | Effect |
|---|---|
| Exposure ×0.5 … ×1.4 (synthetic) | 0 %, but true by construction: rescaling cannot change a log-slope. Real sensor non-linearity needs a real exposure bracket (not done) |
| Exposure ×2 | < 1 % at 200–800; at 0–50 NTU the fit band clips (30–40 % of rings) → `rejected` by QC |
| Extra sensor noise 1–4× (DNG noise model) | ≤ 0.3 % |
| Crop-centre error 5 / 10 / 20 / 40 / 80 px | 2 / 5 / 10 / 24 / 57 %; at 80 px, 0–50 NTU samples jump above the floor. Strongly one-sided: +x/+y shifts change the decay length by 6–10 %, −x/−y by 0–3 %, so the B centroid is not the disc's symmetry centre |

The real risk is geometry, not noise or exposure. In the calibration bursts, the beam position jumped
by 12–86 px between some shots (at 400 NTU, 10 NTU, 200 NTU and 800 NTU), and those jumps account for most of the
spread between shots. At 400 NTU, the two shots taken before the jump predict about 300 NTU and the three after it about 400.

## Calibration protocol (do this to make it real)

1. **Rigid geometry.** Fix the phone, laser and cuvette holder mechanically, with an indexing stop for the
   cuvette so it sits in the same place every time. The calibration bursts moved by up to 86 px, usually after a pause.
2. **Independent samples, not bursts.** At least 3 separate fills per level from fresh dilutions. Re-seat
   the cuvette between shots. Shoot levels in random order. Bursts only measure camera noise, which is
   already negligible.
3. **Formazin standards** (or certified StablCal), checked with a reference turbidimeter on the same day.
4. **More levels where it matters (the single biggest fix):** 0, 25, 50, 75, 100, 150, 200, 300, 400,
   600, 800, 1000 NTU. Today there is nothing between 50 and 200, so the quantifiable floor (~156 NTU) is
   an artefact of that gap.
5. **Exposure.** Keep ISO 40 and f/1.6, and shorten the shutter (for example 1/4000–1/8000 s) or add an ND filter until B
   is unclipped at 0 NTU. Peak-based features and the low range could then become usable. Re-calibrate
   after any change. Exposure is part of the calibration. Shoot one exposure bracket (for example ×0.5, ×1, ×2)
   per level, so exposure robustness is measured on the real sensor and not only simulated.
6. **Dark frame** (laser off, same settings) and **blank frame** (clean water) each session, so background
   and drift can be checked.
7. **Same capture path:** Lightroom iOS RAW (Bayer, not ProRAW), main 26 mm lens, focus fixed at 25 %,
   fixed WB, dark box. The app flags EXIF differences but does not correct them.
8. **Retrain:** add the files to `labels.csv` with a unique `sample_id` per fill, then run steps 1–7.
   Steps 2 and 5 derive the ROI size and the common clip level. If they change, update `PARAMS` in
   `src/pipeline.py` and retrain; the model refuses to load on a mismatch.

## Limitations

- **n = 6 independent samples, one cuvette per level.** The spread between shots underestimates real
  variability (refilling, bubbles, cuvette placement, settling). Every metric above is optimistic.
- **Nothing below ~156 NTU is quantified.** At low turbidity the profile is the camera's own defocus disc
  (10 NTU even looks sharper than 0 NTU), and there is no data between 50 and 200 NTU. The cut-off is
  wherever the learned threshold falls in that gap, so a 100–150 NTU sample would be called "below range".
- **Only 3 quantifiable levels.** The ≥200 fit has 3 points, and each CV fold fits a line through 2.
- **The model design saw the data.** The "≥200 + floor" form was chosen after steps 4–6 showed 0–50 NTU
  was flat, so the CV numbers don't include the cost of that choice.
- **The centre-finder is asymmetric.** The B centroid sits off the disc's symmetry centre (the R and G1
  centroids sit 62 and 19 px away), which makes the features sensitive to the direction of a shift.
- **Geometry-specific.** Decay length is in sensor pixels. A different phone, lens, focus, cuvette,
  distance or laser module invalidates the calibration.
- **The colour leak varies across the frame.** B/R falls from about 0.13 to 0.05 between r = 425 and 800 px. It is
  identical across samples, so it is a fixed property of the camera that the calibration absorbs, but it means the B profile is not the true
  scatter profile. The DNG carries no lens-shading map, only a distortion warp.
- **Clipping:** R is useless, and B is clipped near the centre for 0–50 NTU.
- **Absolute brightness is unreliable** (10 and 50 NTU clip more than 0 NTU), so no intensity-based features are used.
- **Not tested:** colour or absorbing samples (dissolved organics), different particle sizes (formazin
  vs clay vs algae scatter differently), temperature, condensation, scratched cuvettes.
