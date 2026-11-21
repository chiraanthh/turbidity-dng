"""Step 3: radial profile of every image (B and G1 channels), log-y, coloured by NTU."""
import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loaders import DATA_DIR, ROOT, CACHE, load_cached_meta, load_mosaic, normalize, cfa_planes
from preprocess import crop_roi, raw_planes
from profiles import radial_profile

OUT = ROOT / "outputs"
pre = json.loads((CACHE / "step2_preprocess.json").read_text())
P2 = pre["params"]
img = pd.DataFrame(pre["images"])
HALF, R_FULL, NEAR = P2["half"], P2["r_full"], P2["near_clip_dn"]
CHANNELS = ["B", "G1"]
BIN_PX = 5
REF_BAND = (400, 450)   # outside the B saturation radius of every image -> shape normalisation

# ---- 1. profiles ---------------------------------------------------------------------------
prof = {c: [] for c in CHANNELS}
for _, r in img.iterrows():
    m = load_cached_meta(DATA_DIR / r.file)
    mos = load_mosaic(DATA_DIR / r.file)
    P, raw = cfa_planes(normalize(mos, m["black"], m["white"]), m["cfa"]), raw_planes(mos, m["cfa"])
    for c in CHANNELS:
        roi = crop_roi(P[c], r.cy, r.cx, HALF)
        sat = crop_roi(raw[c].astype(np.float32), r.cy, r.cx, HALF) >= m["white"] - NEAR
        rr, mean, std, n = radial_profile(roi, sat, BIN_PX, r_max=R_FULL)
        prof[c].append(mean)
rr = np.asarray(rr)
for c in CHANNELS:
    prof[c] = np.vstack(prof[c])
np.savez(CACHE / "step3_profiles.npz", r=rr, files=img.file.values, ntu=img.ntu.values,
         sample_id=img.sample_id.values, **{c: prof[c] for c in CHANNELS})

warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)   # saturated rings are NaN by design
warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)

# ---- 2. numbers ----------------------------------------------------------------------------
ref = (rr >= REF_BAND[0]) & (rr < REF_BAND[1])
probe = [100, 200, 300, 400, 500, 590]
print(f"ROI half {HALF}, profile out to r_full = {R_FULL} px, ring width {BIN_PX} px")
for c in CHANNELS:
    print(f"\n== {c}: first valid radius (inside it the ring is >1% near-clipped) and mean intensity "
          f"at probe radii, averaged over 5 shots (sd over shots in brackets) ==")
    rows = []
    for (ntu, sid), g in img.groupby(["ntu", "sample_id"]):
        A = prof[c][g.index]
        first = [rr[np.argmax(~np.isnan(a))] for a in A]
        row = dict(ntu=ntu, r_first_valid=f"{np.mean(first):.0f}")
        for p in probe:
            k = np.argmin(abs(rr - p))
            row[f"I@{p}"] = f"{np.nanmean(A[:, k]):.4f} ({np.nanstd(A[:, k]):.4f})"
        q = A[:, np.argmin(abs(rr - 300))] / A[:, np.argmin(abs(rr - 500))]
        row["I@300/I@500"] = f"{np.nanmean(q):.2f} ({np.nanstd(q):.2f})"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

# ---- 3. plot: raw profiles (top) and shape-normalised profiles (bottom) ----------------------
ntus = sorted(img.ntu.unique())
cols = dict(zip(ntus, plt.cm.viridis(np.linspace(0, 0.92, len(ntus)))))
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
for j, c in enumerate(CHANNELS):
    for i, r in img.iterrows():
        a = prof[c][i]
        lab = f"{r.ntu} NTU" if r.shot == 0 else None
        axes[0, j].semilogy(rr, a, color=cols[r.ntu], lw=1, alpha=0.8, label=lab)
        axes[1, j].semilogy(rr, a / np.nanmean(a[ref]), color=cols[r.ntu], lw=1, alpha=0.8, label=lab)
    axes[0, j].set_title(f"{c} plane: azimuthal mean (1.0 = clip). Gaps = ring >1% near-clipped")
    axes[1, j].set_title(f"{c} plane: divided by mean over r = {REF_BAND[0]}-{REF_BAND[1]} px (shape only)")
    axes[1, j].axvspan(*REF_BAND, color="gray", alpha=0.15)
    axes[1, j].set_xlabel("radius from B centre (plane px; x2 for sensor px)")
    for a in axes[:, j]:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
axes[0, 0].set_ylabel("normalised intensity (log)")
axes[1, 0].set_ylabel("relative intensity (log)")
fig.suptitle("Step 3: radial profiles, all 30 images (5 shots per NTU share a colour)")
fig.tight_layout()
fig.savefig(OUT / "step3_radial_profiles.png", dpi=100)
print("\nsaved: outputs/step3_radial_profiles.png, outputs/cache/step3_profiles.npz")
