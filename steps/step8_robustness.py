"""Step 8: robustness of the final model to crop shift, exposure change and extra sensor noise.

One shot per sample (shot 0). Every perturbation goes through pipeline.features_from_plane, i.e. the
exact feature path used by the CLI/app; results are expressed as predicted-NTU change.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loaders import DATA_DIR, ROOT
from pipeline import PARAMS, load_b_plane, sat_threshold, features_from_plane
from preprocess import find_centroid
from model import load_bundle, predict_ntu

OUT = ROOT / "outputs"
bundle = load_bundle(ROOT / "models")
FEAT = bundle["features"][0]
labels = pd.read_csv(ROOT / "labels.csv")
imgs = labels[labels.shot == 0]

SHIFTS = [5, 10, 20, 40, 80]                       # px; burst jumps in step 2 reached ~80 px
SCALES = [0.5, 0.7, 1.4, 2.0]                      # exposure factor
NOISE_K = [1, 2, 4]                                # added variance as multiples of the DNG noise model
NOISE_A, NOISE_B = 9.63074e-05, 9.76561e-07        # DNG NoiseProfile tag: var = A*x + B (normalised units)
rng = np.random.default_rng(0)


def run(B, cy, cx, sat_level, extra_sat=None):
    f, _, _ = features_from_plane(B, cy, cx, sat_level, PARAMS, extra_sat=extra_sat, extras=False)
    p = predict_ntu(bundle, f)
    return f[FEAT], p["ntu"], f["fit_band_valid_frac"]


rows = []
for _, im in imgs.iterrows():
    B, meta = load_b_plane(DATA_DIR / im.file)
    sl = sat_threshold(meta)
    cy, cx = find_centroid(B, **PARAMS["centroid"])
    f0, p0, _ = run(B, cy, cx, sl)
    base = dict(file=im.file, ntu=im.ntu, f0=f0, p0=p0)
    for s in SHIFTS:                                  # 4 directions per magnitude
        for dy, dx in [(s, 0), (-s, 0), (0, s), (0, -s)]:
            f, p, v = run(B, cy + dy, cx + dx, sl)
            rows.append(dict(base, test="shift", level=s, f=f, p=p, band_ok=v))
    orig_sat = B >= sl
    for k in SCALES:
        Bs = np.minimum(B * k, 1.0)                    # brighter: new clipping; dimmer: old clip stays flat
        f, p, v = run(Bs, cy, cx, sl, extra_sat=orig_sat if k < 1 else None)
        rows.append(dict(base, test="exposure", level=k, f=f, p=p, band_ok=v))
    for k in NOISE_K:
        sd = np.sqrt(k * (NOISE_A * np.clip(B, 0, None) + NOISE_B))
        f, p, v = run(B + rng.normal(0, 1, B.shape).astype(np.float32) * sd, cy, cx, sl, extra_sat=orig_sat)
        rows.append(dict(base, test="noise", level=k, f=f, p=p, band_ok=v))
R = pd.DataFrame(rows)
R["dfeat_pct"] = 100 * (R.f / R.f0 - 1)
R["dpred_pct"] = 100 * (R.p / R.p0 - 1)
R["class_flip"] = (R.p0 >= bundle["detection_floor_ntu"]) != (R.p >= bundle["detection_floor_ntu"])
R.to_csv(OUT / "step8_robustness.csv", index=False)

pd.set_option("display.width", 200)
print(f"model: {bundle['name']}  feature: {FEAT}; held-out (LOGO) error 200-800 NTU: "
      f"x/÷ {10**bundle['logo_rmse_log10_above_floor']:.2f}")
print("\n== baseline (shot 0 of each sample) ==")
print(R.drop_duplicates("file")[["file", "ntu", "f0", "p0"]].round(1).to_string(index=False))
hi = R[R.ntu >= 200]
summ = hi.groupby(["test", "level"]).agg(max_abs_dfeat_pct=("dfeat_pct", lambda v: v.abs().max()),
                                         max_abs_dpred_pct=("dpred_pct", lambda v: v.abs().max()),
                                         min_band_ok=("band_ok", "min"))
summ["class_flips_all_samples"] = R.groupby(["test", "level"]).class_flip.sum()
print("\n== worst case over samples >= 200 NTU (%), and floor-class flips over ALL 6 samples ==")
print(summ.round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (t, xl) in zip(axes, [("shift", "centre shift (plane px)"), ("exposure", "exposure factor"),
                              ("noise", "added noise (x DNG noise model variance)")]):
    d = R[R.test == t]
    for (fn, n), g in d.groupby(["file", "ntu"]):
        gg = g.groupby("level").p.agg(["min", "max", "mean"])
        l, = ax.plot(gg.index, gg["mean"], "o-", label=f"{n} NTU")
        ax.fill_between(gg.index, gg["min"], gg["max"], color=l.get_color(), alpha=0.15)
        ax.axhline(g.p0.iloc[0], color=l.get_color(), ls=":", lw=1)
    ax.axhline(bundle["detection_floor_ntu"], color="k", ls="--", lw=1, label="detection floor")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel(xl)
    ax.set_ylabel("predicted NTU")
    ax.set_title(f"{t} (band = min-max, dotted = unperturbed)", fontsize=10)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Step 8: robustness of the final model (one shot per sample)")
fig.tight_layout()
fig.savefig(OUT / "step8_robustness.png", dpi=95)
print("\nsaved: outputs/step8_robustness.csv, outputs/step8_robustness.png")
