"""Step 5: FFT features (Hann window, radial power spectrum, HF ratio, slope, 50% cutoff), B channel.

Two variants per image:
  as-is      : ROI exactly as recorded (sensor clipping differs between images -> artefact risk)
  common-clip: every ROI clipped at the same level T, chosen below the brightest valid ring of
               every image, so all images share an identical, known plateau.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loaders import DATA_DIR, ROOT, CACHE, load_cached_meta, load_mosaic, normalize, cfa_planes
from preprocess import crop_roi
from features import fft_features, radial_power_spectrum, SLOPE_BAND, LF_BAND, HF_BAND, NOISE_BAND, PAD

OUT = ROOT / "outputs"
CH = "B"
pre = json.loads((CACHE / "step2_preprocess.json").read_text())
HALF = pre["params"]["half"]
img = pd.DataFrame(pre["images"])
prof = np.load(CACHE / "step3_profiles.npz", allow_pickle=True)
T = round(0.9 * float(np.min(np.nanmax(prof[CH], axis=1))), 3)   # common clip level
VARIANTS = {"asis": None, "cclip": T}

# ---- 1. spectra + features --------------------------------------------------------------------
rows, spectra = [], {v: [] for v in VARIANTS}
for _, im in img.iterrows():
    m = load_cached_meta(DATA_DIR / im.file)
    roi = crop_roi(cfa_planes(normalize(load_mosaic(DATA_DIR / im.file), m["black"], m["white"]), m["cfa"])[CH],
                   im.cy, im.cx, HALF)
    row = dict(file=im.file, ntu=im.ntu, sample_id=im.sample_id, shot=im.shot)
    for v, cl in VARIANTS.items():
        f, p_mean, _ = radial_power_spectrum(roi, cl)
        spectra[v].append(p_mean)
        row.update({f"{k}_{v}": x for k, x in fft_features(roi, cl).items()})
    rows.append(row)
F = pd.DataFrame(rows)
F.to_csv(OUT / "step5_fft_features.csv", index=False)
np.savez(CACHE / "step5_spectra.npz", f=f, files=img.file.values, **{v: np.vstack(s) for v, s in spectra.items()})

# ---- 2. numbers -------------------------------------------------------------------------------
FEATS = ["hf_ratio", "spec_slope", "f50", "f_noise"]
print(f"ROI {2*HALF}px, Hann window, zero-pad x{PAD}; true resolution {1/(2*HALF):.5f} cycles/px")
print(f"bands (cycles/px): LF {LF_BAND}, HF {HF_BAND}, slope fit {SLOPE_BAND}, noise {NOISE_BAND}")
print(f"common clip level T = {T} (90% of the dimmest image's brightest valid ring)")
g = F.groupby("ntu")
for v in VARIANTS:
    cols = [f"{c}_{v}" for c in FEATS]
    print(f"\n== {v}: per-sample mean (shot sd) ==")
    tab = pd.DataFrame({c: [f"{a:.4g} ({s:.2g})" for a, s in zip(g[c].mean(), g[c].std())] for c in cols},
                       index=g.mean(numeric_only=True).index)
    print(tab.T.to_string())
    means = g[cols].mean()
    print("Spearman rho vs NTU, n=6 sample means (n=30 in brackets):")
    for c in cols:
        r6, p6 = spearmanr(means.index, means[c])
        print(f"  {c:18s} {r6:+.2f}  p={p6:.3f}   ({spearmanr(F.ntu, F[c])[0]:+.2f})")
    print(f"  noise floor: {F[f'noise_floor_{v}'].min():.2e} .. {F[f'noise_floor_{v}'].max():.2e} of total power per bin")

# ---- 3. plots ---------------------------------------------------------------------------------
ntus = sorted(F.ntu.unique())
cmap = dict(zip(ntus, plt.cm.viridis(np.linspace(0, 0.92, len(ntus)))))
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for j, v in enumerate(VARIANTS):
    ax = plt.subplot2grid((2, 4), (0, 2 * j), colspan=2, fig=fig)
    S = np.vstack(spectra[v])
    for i, im in img.iterrows():
        ax.loglog(f[1:], S[i, 1:], color=cmap[im.ntu], lw=0.8, alpha=0.8, label=f"{im.ntu} NTU" if im.shot == 0 else None)
    for b, c, name in [(LF_BAND, "blue", "LF"), (HF_BAND, "red", "HF"), (NOISE_BAND, "gray", "noise")]:
        ax.axvspan(*b, color=c, alpha=0.08, label=name)
    ax.set_title(f"radial power spectrum, {v}" + (f" (T={T})" if VARIANTS[v] else ""))
    ax.set_xlabel("spatial frequency (cycles / plane px)")
    ax.set_xlim(f[1], 0.5)
    ax.set_ylabel("ring-mean power / total non-DC power")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
for ax in axes[0]:
    ax.set_visible(False)
for ax, c in zip(axes[1], FEATS):
    for k, v in enumerate(VARIANTS):
        for n, gg in F.groupby("ntu"):
            x = ntus.index(n) + (k - 0.5) * 0.35 + np.linspace(-0.08, 0.08, len(gg))
            ax.plot(x, gg[f"{c}_{v}"], "o" if k == 0 else "^", ms=4, color=f"C{k}",
                    label=v if n == ntus[0] else None)
    ax.set_xticks(range(len(ntus)), [str(n) for n in ntus])
    ax.set_xlabel("NTU (categorical)")
    ax.set_title(c)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle(f"Step 5: FFT features, {CH} channel, Hann-windowed {2*HALF}x{2*HALF} ROI")
fig.tight_layout()
fig.savefig(OUT / "step5_fft.png", dpi=95)
print("\nsaved: outputs/step5_fft_features.csv, outputs/step5_fft.png, outputs/cache/step5_spectra.npz")
