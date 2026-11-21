"""Step 4: derivative / blur features per image (B channel), with validity flags."""
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
from preprocess import crop_roi, raw_planes
from features import profile_features, image_features, INNER_BAND, HALO_BAND, ANNULUS

OUT = ROOT / "outputs"
CH = "B"
pre = json.loads((CACHE / "step2_preprocess.json").read_text())
HALF, NEAR = pre["params"]["half"], pre["params"]["near_clip_dn"]
img = pd.DataFrame(pre["images"])
prof = np.load(CACHE / "step3_profiles.npz", allow_pickle=True)
r = prof["r"]
assert list(prof["files"]) == list(img.file), "step 3 cache out of sync with step 2 - rerun step 3"

# ---- 1. features -----------------------------------------------------------------------------
rows = []
for i, im in img.iterrows():
    f = dict(file=im.file, ntu=im.ntu, sample_id=im.sample_id, shot=im.shot)
    f.update(profile_features(r, prof[CH][i]))
    m = load_cached_meta(DATA_DIR / im.file)
    mos = load_mosaic(DATA_DIR / im.file)
    roi = crop_roi(cfa_planes(normalize(mos, m["black"], m["white"]), m["cfa"])[CH], im.cy, im.cx, HALF)
    sat = crop_roi(raw_planes(mos, m["cfa"])[CH].astype(np.float32), im.cy, im.cx, HALF) >= m["white"] - NEAR
    f.update(image_features(roi, sat))
    rows.append(f)
F = pd.DataFrame(rows)
F.to_csv(OUT / "step4_features.csv", index=False)

FEATS = ["slope_inner", "slope_halo", "decay_len_inner", "decay_len_halo", "core_halo_ratio", "max_grad",
         "inflection_r", "fwhm", "w10_90", "peak_halo_ratio", "lap_var", "tenengrad"]

# ---- 2. numbers ------------------------------------------------------------------------------
pd.set_option("display.width", 220)
print(f"channel {CH}; flank band {INNER_BAND}, halo band {HALO_BAND}, 2-D annulus {ANNULUS} px")
print(f"images with unclipped peak: {int(F.peak_ok.sum())}/30 "
      f"(NTU: {sorted(F[F.peak_ok].ntu.unique().tolist())})")
print("\n== per-sample mean (shot sd) ==")
g = F.groupby("ntu")
tab = pd.DataFrame({c: [f"{a:.4g} ({s:.2g})" if np.isfinite(a) else "n/a"
                        for a, s in zip(g[c].mean(), g[c].std())] for c in FEATS}, index=g.mean(numeric_only=True).index)
print(tab.T.to_string())

print("\n== Spearman rho vs NTU. 'n=6' uses sample means -> the honest count; "
      "n=30 treats burst shots as independent (they are not) ==")
means = g[FEATS].mean()
for c in FEATS:
    v = F[c].notna()
    if F.loc[v, "ntu"].nunique() < 3:
        print(f"{c:16s}  defined for only {F.loc[v, 'ntu'].nunique()} NTU level(s) -> not usable")
        continue
    r30 = spearmanr(F.ntu[v], F[c][v])[0]
    mv = means[c].dropna()
    r6, p6 = spearmanr(mv.index, mv.values)
    print(f"{c:16s}  rho(n=30) {r30:+.2f}   rho(n={len(mv)}) {r6:+.2f}  p={p6:.3f}")

# ---- 3. plot ---------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
ntus = sorted(F.ntu.unique())
xpos = {n: k for k, n in enumerate(ntus)}           # categorical x: only 6 levels
for ax, c in zip(axes.flat, FEATS):
    for n, gg in F.groupby("ntu"):
        x = xpos[n] + np.linspace(-0.15, 0.15, len(gg))
        ax.plot(x, gg[c], "o", ms=4, alpha=0.7, color="C0")
        if gg[c].notna().any():
            ax.errorbar(xpos[n] + 0.3, gg[c].mean(), gg[c].std(), fmt="s", color="k", ms=5, capsize=3)
    ax.set_xticks(range(len(ntus)), [str(n) for n in ntus])
    ax.set_title(c + ("  (needs unclipped peak)" if c in ("inflection_r", "fwhm", "w10_90", "peak_halo_ratio") else ""),
                 fontsize=10)
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("NTU (categorical)")
fig.suptitle(f"Step 4: derivative features, {CH} channel. Dots = 5 burst shots, black = mean ± sd")
fig.tight_layout()
fig.savefig(OUT / "step4_features.png", dpi=95)
print("\nsaved: outputs/step4_features.csv, outputs/step4_features.png")
