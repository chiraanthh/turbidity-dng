"""Step 2: compare channels, find the spot centre on B, crop a fixed ROI, clip checks, G/R and B/R ratios."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loaders import DATA_DIR, ROOT, CACHE, load_cached_meta, load_mosaic, normalize, cfa_planes
from preprocess import CHANNELS, find_centroid, crop_roi, raw_planes

OUT = ROOT / "outputs"
labels = pd.read_csv(ROOT / "labels.csv")

CENTRE_CH = "B"          # least-clipped channel -> centroid
SIGMA, FRAC, FLOOR_PCT = 10, 0.5, 5
NEAR_CLIP_DN = 64        # also count pixels within 64 DN of white (sensor response bends before hard clip)
RATIO_LO, RATIO_HI = 0.2, 0.9   # R range used for G/R, B/R (unclipped, well above noise)
MAX_NAN = 0.01           # ROI may lose at most 1% of its pixels off-frame, in the worst image
B_FLAG_PCT = 1.0         # flag image if > 1% of B pixels in ROI are clipped

# ---- 1. per image: centroids on every channel (diagnostic), whole-frame clip, ratios ----------
rows, cache_planes = [], {}
for _, lab in labels.iterrows():
    m = load_cached_meta(DATA_DIR / lab.file)
    mos = load_mosaic(DATA_DIR / lab.file)
    raw = raw_planes(mos, m["cfa"])
    P = cfa_planes(normalize(mos, m["black"], m["white"]), m["cfa"])
    r = dict(file=lab.file, ntu=lab.ntu, sample_id=lab.sample_id, shot=lab.shot)
    for c in CHANNELS:
        cen = find_centroid(P[c], SIGMA, FRAC, FLOOR_PCT)
        r[f"cy_{c}"], r[f"cx_{c}"] = cen if cen else (np.nan, np.nan)
        r[f"clip_{c}"] = float((raw[c] >= m["white"]).mean())
        r[f"near_{c}"] = float((raw[c] >= m["white"] - NEAR_CLIP_DN).mean())
    # sensitivity ratios: per-pixel, only where R is linear and every channel is unclipped
    ok = (P["R"] > RATIO_LO) & (P["R"] < RATIO_HI) & (P["G1"] < 1) & (P["G2"] < 1) & (P["B"] < 1)
    for c in ["G1", "G2", "B"]:
        r[f"{c}/R"] = float(np.median(P[c][ok] / P["R"][ok]))
    r["G1/G2"] = float(np.median(P["G1"][ok] / P["G2"][ok]))
    r["ratio_npx"] = int(ok.sum())
    r["cy"], r["cx"] = r[f"cy_{CENTRE_CH}"], r[f"cx_{CENTRE_CH}"]
    rows.append(r)
    cache_planes[lab.file] = (P, raw, m)
df = pd.DataFrame(rows)
H, W = next(iter(cache_planes.values()))[0]["R"].shape

# ---- 2. choose ROI half-size: largest size whose worst-case NaN padding stays <= MAX_NAN ----
# Bigger ROI keeps more halo (matters at high NTU); NaN padding differs per image and would make
# large-radius features track spot position. Allow a sliver of padding, no more.
edge = np.minimum.reduce([df.cy, df.cx, H - df.cy, W - df.cx])
R_FULL = int(np.floor(edge.min()))   # radius that is fully inside the frame for EVERY image
print(f"plane size {H}x{W}; closest centroid-to-edge distance over all 30 images: {R_FULL} px")
print("\nhalf-size  max NaN frac  #images with NaN")
HALF = 0
for h in range(300, 801, 50):
    nanf = [np.isnan(crop_roi(cache_planes[r.file][0]["B"], r.cy, r.cx, h)).mean() for _, r in df.iterrows()]
    print(f"{h:9d}  {max(nanf):12.3f}  {sum(f > 0 for f in nanf):5d}")
    if max(nanf) <= MAX_NAN:
        HALF = h
print(f"-> chosen HALF = {HALF} (ROI {2*HALF}x{2*HALF} plane px = {4*HALF}x{4*HALF} sensor px); "
      f"radial features should stop at r_full = {R_FULL} px")

# ---- 3. ROI clip fractions + NaN fraction ------------------------------------------------
for i, r in df.iterrows():
    P, raw, m = cache_planes[r.file]
    for c in CHANNELS:
        roi_raw = crop_roi(raw[c].astype(np.float32), r.cy, r.cx, HALF)
        v = roi_raw[~np.isnan(roi_raw)]
        df.loc[i, f"roiclip_{c}"] = float((v >= m["white"]).mean())
    df.loc[i, "roi_nan"] = float(np.isnan(crop_roi(P["B"], r.cy, r.cx, HALF)).mean())
df["flag_B_sat"] = df.roiclip_B * 100 > B_FLAG_PCT

pd.set_option("display.width", 200)
pct = lambda s: (100 * s).round(2)
print("\n== clipped pixel % (raw DN >= white=4095), whole plane | inside ROI ==")
t = df[["file", "ntu"]].copy()
for c in CHANNELS:
    t[f"{c}"] = pct(df[f"clip_{c}"])
for c in CHANNELS:
    t[f"roi_{c}"] = pct(df[f"roiclip_{c}"])
t["near_B"] = pct(df.near_B)
t["roi_nan"] = df.roi_nan.round(3)
t["flag"] = df.flag_B_sat.map({True: "SAT", False: ""})
print(t.to_string(index=False))

print("\n== centroid (plane px) per channel; offsets vs B ==")
for c in ["R", "G1"]:
    dy, dx = df[f"cy_{c}"] - df.cy_B, df[f"cx_{c}"] - df.cx_B
    print(f"{c}-B: dy mean {dy.mean():+.1f} (sd {dy.std():.1f}), dx mean {dx.mean():+.1f} (sd {dx.std():.1f})")

print("\n== burst jitter (5 shots, same cuvette) vs between-sample spread, B centroid ==")
g = df.groupby(["ntu", "sample_id"])
j = g.agg(cy=("cy", "mean"), cx=("cx", "mean"), sd_cy=("cy", "std"), sd_cx=("cx", "std"), n=("cy", "size"))
print(j.round(2).to_string())
print(f"between-sample sd: cy {j.cy.std():.1f} px, cx {j.cx.std():.1f} px")

print("\n== sensitivity ratios (median per-pixel, R in [0.2,0.9], nothing clipped) ==")
print(df.groupby("ntu")[["G1/R", "G2/R", "B/R", "G1/G2", "ratio_npx"]].agg(["mean", "std"]).round(4).to_string())
print(f"ALL: G1/R {df['G1/R'].mean():.4f}±{df['G1/R'].std():.4f}  B/R {df['B/R'].mean():.4f}±{df['B/R'].std():.4f}"
      f"  G1/G2 {df['G1/G2'].mean():.4f}")

# ---- 4. ratio vs radius, split by R level (separates "sensor nonlinearity" from "position") ----
# If the ratio changes with R level at fixed radius -> response is nonlinear near clip.
# If it changes with radius at fixed R level -> colour response varies across the frame (optics/sensor).
redges = np.arange(0, 900, 50)
RBINS = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.9)]
ratio_r = {}      # ntu -> {rbin: (G1/R list, B/R list)}
for f in df.groupby("ntu").head(1).file:
    P, _, _ = cache_planes[f]
    r = df[df.file == f].iloc[0]
    yy, xx = np.indices(P["R"].shape)
    rad = np.hypot(yy - r.cy, xx - r.cx)
    unclip = (P["G1"] < 1) & (P["B"] < 1)
    ratio_r[int(r.ntu)] = {}
    for lo, hi in RBINS:
        ok = unclip & (P["R"] > lo) & (P["R"] < hi)
        gr, br = [], []
        for a, b in zip(redges[:-1], redges[1:]):
            k = ok & (rad >= a) & (rad < b)
            gr.append(np.median(P["G1"][k] / P["R"][k]) if k.sum() > 200 else np.nan)
            br.append(np.median(P["B"][k] / P["R"][k]) if k.sum() > 200 else np.nan)
        ratio_r[int(r.ntu)][(lo, hi)] = (gr, br)
rc = 0.5 * (redges[:-1] + redges[1:])
print("\n== B/R vs radius, per R-level bin (first image of each NTU; '-' = <200 px) ==")
for ntu, d in sorted(ratio_r.items()):
    for rb, (gr, br) in d.items():
        print(f"{ntu:4d} NTU R in {rb}: " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in br))
print("radius bins: " + " ".join(f"{int(v):5d}" for v in rc))

# ---- 4b. saturation radius: how far out from the centre do near-clip pixels reach? -----------
for i, r in df.iterrows():
    P, raw, m = cache_planes[r.file]
    for c in CHANNELS:
        ys, xs = np.nonzero(raw[c] >= m["white"] - NEAR_CLIP_DN)
        df.loc[i, f"rsat_{c}"] = float(np.hypot(ys - r.cy, xs - r.cx).max()) if len(ys) else 0.0
print("\n== r_sat (plane px): max distance from B centre of any pixel >= white-64 ==")
print(df.groupby("ntu")[[f"rsat_{c}" for c in CHANNELS]].max().round(0).to_string())

# ---- 5. plots -----------------------------------------------------------------------------
first = df.groupby("ntu").head(1).sort_values("ntu")
fig, axes = plt.subplots(2, 6, figsize=(24, 8.5))
for jx, (_, r) in enumerate(first.iterrows()):
    P = cache_planes[r.file][0]
    for row, c in enumerate(["B", "R"]):
        ax = axes[row, jx]
        ax.imshow(np.sqrt(np.clip(P[c], 0, 1)), cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(P[c] < 1, np.ones_like(P[c])), cmap="autumn", alpha=0.8)  # clipped = red/yellow
        ax.add_patch(plt.Rectangle((r.cx - HALF, r.cy - HALF), 2 * HALF, 2 * HALF, fill=False, ec="cyan", lw=1.5))
        ax.plot(r.cx_B, r.cy_B, "c+", ms=14, mew=2, label="B centre (used)")
        ax.plot(r.cx_R, r.cy_R, "rx", ms=10, mew=2, label="R centre")
        ax.plot(r.cx_G1, r.cy_G1, "gx", ms=10, mew=2, label="G1 centre")
        ax.set_title(f"{c} plane, {r.ntu} NTU ({r.file})\nclipped in frame {100*r[f'clip_{c}']:.2f}%", fontsize=10)
        ax.axis("off")
axes[0, 0].legend(loc="lower left", fontsize=8)
fig.suptitle(f"Step 2: B (top) and R (bottom) planes, sqrt stretch. Clipped pixels overlaid in orange. "
             f"Cyan box = fixed ROI {2*HALF}x{2*HALF} centred on B centroid")
fig.tight_layout()
fig.savefig(OUT / "step2_roi.png", dpi=90)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
x = np.arange(len(df))
for k, c in enumerate(CHANNELS):
    axes[0].bar(x + (k - 1.5) * 0.2, 100 * df[f"roiclip_{c}"], 0.2, label=c)
axes[0].axhline(B_FLAG_PCT, color="k", ls="--", lw=1, label=f"B flag {B_FLAG_PCT}%")
axes[0].set_xticks(x, [f"{f[4:8]}\n{n}" for f, n in zip(df.file, df.ntu)], fontsize=7)
axes[0].set_ylabel("% clipped pixels in ROI (raw DN >= 4095)")
axes[0].set_yscale("symlog", linthresh=0.1)
axes[0].set_title("Clipped fraction inside ROI, per image & channel (x: file / NTU)")
axes[0].legend()
styles = {RBINS[0]: ":", RBINS[1]: "--", RBINS[2]: "-"}
cols = dict(zip(sorted(ratio_r), plt.cm.viridis(np.linspace(0, 0.9, len(ratio_r)))))
for ntu, d in sorted(ratio_r.items()):
    for rb, (gr, br) in d.items():
        axes[1].plot(rc, br, styles[rb], color=cols[ntu], marker=".",
                     label=f"{ntu} NTU" if rb == RBINS[2] else None)
for rb, ls in styles.items():
    axes[1].plot([], [], ls, color="gray", label=f"R in {rb}")
axes[1].axvline(HALF, color="k", ls=":", lw=1)
axes[1].set_yscale("log")
axes[1].set_xlabel("radius from B centre (plane px)")
axes[1].set_ylabel("median B/R")
axes[1].set_title("B/R vs radius, split by R level. Flat = pure linear leak")
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "step2_clip_ratio.png", dpi=100)

# ---- 6. cache -----------------------------------------------------------------------------
params = dict(centre_channel=CENTRE_CH, sigma=SIGMA, frac=FRAC, floor="percentile", floor_pct=FLOOR_PCT,
              half=HALF, roi_size_plane_px=2 * HALF, r_full=R_FULL, max_nan=MAX_NAN, black=528.0, white=4095.0, clip_rule="raw DN >= white",
              ratio_R_range=[RATIO_LO, RATIO_HI], sat_flag_rule=f"roiclip_B > {B_FLAG_PCT}%", near_clip_dn=NEAR_CLIP_DN,
              rsat_B_max=float(df.rsat_B.max()), opcodes="OpcodeList3 WarpRectilinear only; no GainMap",
              G1_over_R=float(df["G1/R"].mean()), G2_over_R=float(df["G2/R"].mean()), B_over_R=float(df["B/R"].mean()))
(CACHE / "step2_preprocess.json").write_text(json.dumps(
    dict(params=params, images=json.loads(df.to_json(orient="records"))), indent=1))
df.to_csv(OUT / "step2_preprocess.csv", index=False)
print(f"\nsaved: outputs/step2_roi.png, outputs/step2_clip_ratio.png, outputs/step2_preprocess.csv, "
      f"outputs/cache/step2_preprocess.json\nflagged (B >{B_FLAG_PCT}% clipped in ROI): {df.flag_B_sat.sum()} / {len(df)}")
