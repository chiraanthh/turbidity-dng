"""Step 1: load every DNG, report metadata, show one image per NTU level, run a noise check."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loaders import DATA_DIR, ROOT, load_cached_meta, load_mosaic, normalize, cfa_planes, demosaic_linear

OUT = ROOT / "outputs"
labels = pd.read_csv(ROOT / "labels.csv")

# ---- 1. metadata for every file -------------------------------------------------
rows = []
for _, lab in labels.iterrows():
    p = DATA_DIR / lab.file
    m = load_cached_meta(p)
    mos = load_mosaic(p)                                 # also fills the cache
    m.update(ntu=lab.ntu, sample_id=lab.sample_id,
             raw_max=int(mos.max()), clipped_px=int((mos >= m["white"]).sum()))
    rows.append(m)
meta = pd.DataFrame(rows)
meta.to_csv(OUT / "step1_metadata.csv", index=False)

print("\n== counts per NTU ==")
print(meta.groupby("ntu").size().to_string())
print("\n== unique values across ALL images (should be 1 each if settings were fixed) ==")
for c in ["raw_type", "cfa", "black", "white", "bit_depth", "width", "height",
          "exposure", "iso", "fnumber", "focal35", "lens", "software"]:
    print(f"{c:10s}: {meta[c].unique().tolist()}")
print("\n== per-image raw max and clipped pixel count (all channels, raw DN >= white) ==")
print(meta[["file", "ntu", "datetime", "raw_max", "clipped_px"]].to_string(index=False))

# ---- 2. grid: one sample per NTU level ---------------------------------------
first = meta.groupby("ntu").head(1).sort_values("ntu")
fig, axes = plt.subplots(2, len(first), figsize=(4 * len(first), 7))
for j, (_, m) in enumerate(first.iterrows()):
    norm = normalize(load_mosaic(DATA_DIR / m.file), m.black, m.white)
    R = cfa_planes(norm, m.cfa)["R"]
    # top: full red plane, simple viewing stretch (sqrt) -- display only
    axes[0, j].imshow(np.sqrt(np.clip(R, 0, 1)), cmap="gray", vmin=0, vmax=1)
    axes[0, j].set_title(f"{m.ntu} NTU  ({m.file})")
    # bottom: 400x400 crop around the brightest (smoothed) point
    from scipy.ndimage import uniform_filter
    cy, cx = np.unravel_index(np.argmax(uniform_filter(R, 25)), R.shape)
    h = 200
    crop = R[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h]
    axes[1, j].imshow(np.sqrt(np.clip(crop, 0, 1)), cmap="inferno", vmin=0, vmax=1)
    axes[1, j].set_title(f"brightest region @({cy},{cx}), max={crop.max():.2f}")
for a in axes.flat:
    a.axis("off")
fig.suptitle("Red CFA plane (raw, no demosaic), sqrt stretch for viewing. 1.0 = sensor clip")
fig.tight_layout()
fig.savefig(OUT / "step1_grid.png", dpi=110)

# ---- 3. noise check: is this real single-frame RAW or processed/fused? -------
# Real Bayer RAW: dark-region noise is independent pixel to pixel -> lag-1 autocorrelation ~ 0.
# Demosaiced / denoised / multi-frame data: noise is spatially smoothed -> autocorrelation >> 0.
m = meta.iloc[1]
norm = normalize(load_mosaic(DATA_DIR / m.file), m.black, m.white)
G = cfa_planes(norm, m.cfa)["G1"]
patch = G[50:350, 50:350]                       # a corner, assumed dark (box covered)
d = patch - patch.mean()
ac = (d[:, :-1] * d[:, 1:]).mean() / d.var()
print(f"\n== noise check on {m.file} G1 corner patch ==")
print(f"mean={patch.mean()*(m.white-m.black):.2f} DN above black, std={patch.std()*(m.white-m.black):.2f} DN, "
      f"lag-1 autocorr={ac:.3f}")
raw_corner = load_mosaic(DATA_DIR / m.file)[100:700:2, 100:700:2]
print(f"raw DN values in that corner: min={raw_corner.min()} max={raw_corner.max()} "
      f"distinct={len(np.unique(raw_corner))}")

# demosaiced vs raw-plane check (same image): demosaic must stay linear and not shift levels
rgb = demosaic_linear(DATA_DIR / m.file).astype(np.float32)
Rp = cfa_planes(load_mosaic(DATA_DIR / m.file).astype(np.float32), m.cfa)["R"]
print(f"\n== demosaic sanity == demosaiced R mean={rgb[...,0].mean():.1f} DN, raw R-plane mean "
      f"(before black subtraction)={Rp.mean():.1f} DN, after={Rp.mean()-m.black:.1f} DN")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(raw_corner.ravel(), bins=np.arange(raw_corner.min(), raw_corner.max() + 2) - 0.5)
ax[0].axvline(m.black, color="r", ls="--", label=f"black level {m.black:.0f}")
ax[0].set(title="Dark-corner raw DN histogram", xlabel="raw DN", ylabel="pixels"); ax[0].legend()
lags = range(0, 8)
acs = [1.0] + [((d[:, :-k] * d[:, k:]).mean() / d.var()) for k in lags[1:]]
ax[1].plot(list(lags), acs, "o-"); ax[1].axhline(0, color="k", lw=0.5)
ax[1].set(title="Noise autocorrelation vs pixel lag (G plane)", xlabel="lag (plane px)", ylabel="autocorr")
fig.tight_layout()
fig.savefig(OUT / "step1_noise_check.png", dpi=110)
print("\nsaved outputs/step1_grid.png, outputs/step1_noise_check.png, outputs/step1_metadata.csv")
