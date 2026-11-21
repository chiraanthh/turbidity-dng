"""Build every presentation table from the saved step outputs, then fill docs/explanation_template.md
-> explanation.md. Rerun after retraining and the document stays consistent with the code.

Outputs: outputs/tables/<name>.csv, <name>.md, png/<name>.png (slide-ready) and explanation.md.
"""
import json
import re
import sys
from pathlib import Path

import exifread
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loaders import DATA_DIR, CACHE      # noqa: E402
from model import to_ntu                 # noqa: E402

OUT = ROOT / "outputs"
TAB = OUT / "tables"
(TAB / "png").mkdir(parents=True, exist_ok=True)
NTUS = [0, 10, 50, 200, 400, 800]

labels = pd.read_csv(ROOT / "labels.csv")
meta1 = pd.read_csv(OUT / "step1_metadata.csv")
pre = json.loads((CACHE / "step2_preprocess.json").read_text())
s2 = pd.read_csv(OUT / "step2_preprocess.csv")
f4 = pd.read_csv(OUT / "step4_features.csv")
f5 = pd.read_csv(OUT / "step5_fft_features.csv")
s6 = pd.read_csv(OUT / "step6_feature_stats.csv")
sel6 = json.loads((CACHE / "step6_selection.json").read_text())
m7 = pd.read_csv(OUT / "step7_metrics.csv")
lv7 = pd.read_csv(OUT / "step7_per_level.csv")
p7 = pd.read_csv(OUT / "step7_predictions.csv")
r8 = pd.read_csv(OUT / "step8_robustness.csv")
prof = np.load(CACHE / "step3_profiles.npz", allow_pickle=True)
bundle = joblib.load(ROOT / "models" / "turbidity_model.joblib")
params = json.loads((ROOT / "models" / "params.json").read_text())
FLOOR = bundle["detection_floor_ntu"]

TABLES, VALUES = {}, {}


def pm(mu, sd, fmt="{:.3g}"):
    return f"{fmt.format(mu)} ± {fmt.format(sd)}" if np.isfinite(mu) else "n/a"


def add(name, df, title):
    TABLES[name] = (df, title)


# ---- 1. acquisition settings ------------------------------------------------------------------
m0 = meta1.iloc[0]
uniq = lambda c: "identical on all 30" if meta1[c].nunique() == 1 else f"{meta1[c].nunique()} values"
add("setup", pd.DataFrame([
    ("Camera", "iPhone 16, main (wide) lens, 26 mm equiv.", m0.lens),
    ("Capture app / format", "Lightroom iOS, DNG", f"raw_type={m0.raw_type} (Bayer, not ProRAW)"),
    ("Sensor data", f"{int(m0.bit_depth)}-bit, CFA {m0.cfa}", f"black {m0.black:.0f}, white {m0.white:.0f} DN"),
    ("ISO / shutter / aperture", f"{m0.iso} / {m0.exposure} s / f/1.6", uniq("exposure")),
    ("Focus / white balance", "manual 25 %, fixed WB", "set in app"),
    ("Image size", f"{int(m0.width)}×{int(m0.height)} px (planes {int(m0.width)//2}×{int(m0.height)//2})", ""),
    ("Light source", "red laser + dot-generator cap", "wavelength not measured"),
    ("Geometry", "camera looks straight down the beam through the cuvette", "dark box"),
], columns=["Item", "Setting", "Note"]), "Acquisition setup")

# ---- 2. dataset -------------------------------------------------------------------------------
times = {}
for f in labels.file:
    with open(DATA_DIR / f, "rb") as fh:
        t = exifread.process_file(fh, details=False)
    times[f] = pd.to_datetime(str(t["EXIF DateTimeOriginal"]), format="%Y:%m:%d %H:%M:%S") + \
        pd.Timedelta(milliseconds=int(str(t.get("EXIF SubSecTimeOriginal", "0"))))
labels["time"] = labels.file.map(times)
rows = []
for n in NTUS:
    g = labels[labels.ntu == n].sort_values("time")
    gaps = g.time.diff().dt.total_seconds().dropna()
    rows.append((n, g.sample_id.iloc[0], f"{g.file.iloc[0][:8]}–{g.file.iloc[-1][4:8]}", len(g),
                 g.time.iloc[0].strftime("%H:%M:%S"), f"{gaps.min():.1f}–{gaps.max():.1f}"))
add("dataset", pd.DataFrame(rows, columns=["NTU", "sample_id", "files", "shots", "first shot",
                                           "gap between shots (s)"]), "Dataset (6 independent samples × 5 burst shots)")
VALUES["n_images"] = len(labels)
VALUES["n_samples"] = labels.sample_id.nunique()

# ---- 3. saturation + sensitivity ratios ---------------------------------------------------------
g2 = s2.groupby("ntu")
sat = pd.DataFrame({"NTU": NTUS})
for c in ["R", "G1", "G2", "B"]:
    sat[f"{c} clipped, frame (%)"] = [100 * g2.get_group(n)[f"clip_{c}"].mean() for n in NTUS]
sat["B clipped, ROI (%)"] = [100 * g2.get_group(n).roiclip_B.mean() for n in NTUS]
sat["B near-clip (%)"] = [100 * g2.get_group(n).near_B.mean() for n in NTUS]
sat["B saturated out to r (px)"] = [round(g2.get_group(n).rsat_B.max()) for n in NTUS]
sat["flagged (B ROI > 1 %)"] = [f"{int(g2.get_group(n).flag_B_sat.sum())}/5" for n in NTUS]
add("saturation", sat.round(2), "Clipping per channel (raw DN ≥ 4095), mean of 5 shots")
rat = pd.DataFrame({"NTU": NTUS})
for c in ["G1/R", "G2/R", "B/R", "G1/G2"]:
    rat[c] = [pm(g2.get_group(n)[c].mean(), g2.get_group(n)[c].std(), "{:.4f}") for n in NTUS]
add("ratios", rat, "Channel sensitivity to the red laser (median per-pixel ratio, R in 0.2–0.9)")
VALUES["B_over_R"] = f"{s2['B/R'].mean():.3f}"
VALUES["G_over_R"] = f"{s2['G1/R'].mean():.3f}"

# ---- 4. geometry within bursts --------------------------------------------------------------------
geo = s2.merge(labels[["file", "time"]], on="file").sort_values("time")
geo["jump_px"] = np.hypot(geo.groupby("sample_id").cy.diff(), geo.groupby("sample_id").cx.diff())
geo["gap_s"] = geo.groupby("sample_id").time.diff().dt.total_seconds()
gt = geo[["file", "ntu", "shot", "cy", "cx", "jump_px", "gap_s"]].copy()
gt.columns = ["file", "NTU", "shot", "centre y (px)", "centre x (px)", "jump vs previous (px)", "gap (s)"]
add("geometry", gt.round(1).fillna("–"), "Spot centre per shot (B plane px): bursts are NOT identical")
geo_s = geo.groupby("ntu").agg(max_jump=("jump_px", "max"), sd_y=("cy", "std"), sd_x=("cx", "std")).reindex(NTUS)
geo_s["type"] = np.where(geo_s.max_jump > 10, "discrete jump(s)", "smooth drift")
geo_s = geo_s.reset_index().round(1)
geo_s.columns = ["NTU", "largest jump (px)", "sd y (px)", "sd x (px)", "behaviour"]
add("geometry_summary", geo_s, "Within-burst movement of the spot, per sample")
VALUES["max_jump"] = f"{geo.jump_px.max():.0f}"
jumps = geo.jump_px[geo.jump_px > 10]
VALUES["jump_range"] = f"{jumps.min():.0f}–{jumps.max():.0f}"

# ---- 5. radial profile -------------------------------------------------------------------------
r, B = prof["r"], prof["B"]
pr = []
for n in NTUS:
    A = B[prof["ntu"] == n]
    first = np.mean([r[np.argmax(np.isfinite(a))] for a in A])
    row = [n, f"{first:.0f}"]
    for rad in (200, 300, 400, 500, 590):
        v = A[:, np.argmin(abs(r - rad))]
        row.append("clipped" if np.isnan(v).all() else f"{np.nanmean(v):.3f}")
    q = A[:, np.argmin(abs(r - 300))] / A[:, np.argmin(abs(r - 500))]
    row.append(pm(np.nanmean(q), np.nanstd(q), "{:.2f}"))
    pr.append(row)
add("profile", pd.DataFrame(pr, columns=["NTU", "first unclipped r (px)", "I(200)", "I(300)", "I(400)", "I(500)",
                                         "I(590)", "I(300)/I(500)"]),
    "B-channel radial profile (normalised intensity, 1 = clip), mean of 5 shots")

# ---- 6. features per level -----------------------------------------------------------------------
F = f4.merge(f5, on=["file", "ntu", "sample_id", "shot"])
FEATS = [("decay_len_inner", "decay length, flank (px)", "{:.1f}"),
         ("decay_len_halo", "decay length, halo (px)", "{:.1f}"),
         ("core_halo_ratio", "core/halo I(300)/I(500)", "{:.2f}"),
         ("tenengrad", "Tenengrad (norm.)", "{:.4f}"),
         ("lap_var", "Laplacian var (norm.)", "{:.2e}"),
         ("spec_slope_cclip", "spectral slope", "{:.2f}"),
         ("hf_ratio_asis", "HF ratio (as-is)", "{:.2e}"),
         ("fwhm", "FWHM (px)", "{:.0f}")]
gF = F.groupby("ntu")
ft = pd.DataFrame({"feature": [lab for _, lab, _ in FEATS]})
for n in NTUS:
    ft[f"{n} NTU"] = [pm(gF.get_group(n)[c].mean(), gF.get_group(n)[c].std(), fmt) for c, _, fmt in FEATS]
add("features", ft, "Candidate features, mean ± sd over 5 shots (B channel)")

# ---- 7. feature ranking + redundancy --------------------------------------------------------------
rk = s6.copy()
rk["role"] = np.where(rk.feature.isin(sel6["shortlist"]), "SELECTED",
                      np.where(rk.feature.isin(sel6["cross_checks"]), "cross-check",
                               np.where(rk.confound.notna(), "excluded (confound)", "redundant")))
rk = rk[["feature", "rho6", "p6", "ordered_pairs", "sep 50-200", "sep 200-400", "sep 400-800", "cv_shots", "role",
         "confound"]].fillna("")
rk["cv_shots"] = (100 * rk.cv_shots).round(1).astype(str) + " %"
rk.columns = ["feature", "Spearman ρ (n=6)", "p", "adjacent pairs in order", "sep 50→200", "sep 200→400",
              "sep 400→800", "shot CV", "role", "why excluded"]
rk[["sep 50→200", "sep 200→400", "sep 400→800"]] = rk[["sep 50→200", "sep 200→400", "sep 400→800"]].astype(float).round(1)
rk[["Spearman ρ (n=6)", "p"]] = rk[["Spearman ρ (n=6)", "p"]].astype(float).round(3)
add("ranking", rk, "Feature ranking (sep = |Δmean| / combined shot sd; >3 ≈ resolvable, optimistic)")
clean = [c for c in s6.feature if c not in sel6["excluded_confounded"]]
corr = F.groupby("ntu")[clean].mean().corr().round(3)
add("redundancy", corr.reset_index().rename(columns={"index": "feature"}),
    "Pearson r between the 6 sample means (all ≥ 0.95 in magnitude ⇒ one underlying measurement)")

# ---- 8. models -----------------------------------------------------------------------------------
mt = m7.copy()
mt.columns = ["model", "MAE (NTU)", "RMSE (NTU)", "R²", "RMSE log10(NTU+1)"]
mt["chosen"] = np.where(mt.model == bundle["name"], "✔", "")
mt = mt.round({"R²": 2, "RMSE log10(NTU+1)": 2})
mt["MAE (NTU)"], mt["RMSE (NTU)"] = mt["MAE (NTU)"].round().astype(int), mt["RMSE (NTU)"].round().astype(int)
add("models", mt, "Leave-one-sample-out CV (6 folds; each fold holds out a whole NTU level)")
pl = lv7.copy()
fl = pl.model == bundle["name"]
for n in NTUS:
    col = str(n)
    pl[col] = [("below range" if (is_f and v < FLOOR) else f"{v:.0f}") for v, is_f in zip(pl[col], fl)]
pl.columns = ["model"] + [f"true {n}" for n in NTUS]
add("per_level", pl, "Mean held-out prediction per level (NTU)")

pp = p7.copy()
pp["status (held out)"] = np.where(pp.logo < FLOOR, "below range", "quantified")
pp["correct class"] = np.where((pp.ntu >= 200) == (pp.logo >= FLOOR), "yes", "NO")
fmt_ntu = lambda v: "below range" if v < FLOOR else f"{v:.0f}"
pp["in-sample"] = pp.in_sample.map(fmt_ntu)
pp["held-out (LOGO)"] = pp.logo.map(fmt_ntu)
pp["held-out error"] = [("–" if t < 200 or p < FLOOR else f"{100*(p/t-1):+.0f} %") for t, p in zip(pp.ntu, pp.logo)]
pp = pp[["file", "ntu", "in-sample", "held-out (LOGO)", "held-out error", "correct class"]]
pp.columns = ["file", "true NTU", "fit on all data", "held-out (LOGO)", "held-out error", "correct class"]
add("predictions", pp, "Final model, every image")

q = p7[p7.ntu >= 200]
okq = q[q.logo >= FLOOR]
VALUES.update(
    model_name=bundle["name"], feature=bundle["features"][0], floor=f"{FLOOR:.0f}",
    factor=f"{10**bundle['logo_rmse_log10_above_floor']:.2f}",
    factor_ok=f"{10**np.sqrt(np.mean(np.log10(okq.logo/okq.ntu)**2)):.2f}",
    low_acc=f"{100*bundle['logo_low_class_acc']:.0f}", hi_acc=f"{100*bundle['logo_high_class_acc']:.0f}",
    n_hi_ok=f"{len(okq)}/{len(q)}",
    mae=f"{m7.set_index('model').loc[bundle['name'], 'MAE']:.0f}",
    r2=f"{m7.set_index('model').loc[bundle['name'], 'R2']:.2f}",
    logo_200=f"{lv7.set_index('model').loc[bundle['name'], '200']:.0f}",
    logo_400=f"{lv7.set_index('model').loc[bundle['name'], '400']:.0f}",
    logo_800=f"{lv7.set_index('model').loc[bundle['name'], '800']:.0f}",
)
lr = bundle["model"].lr_
VALUES["equation"] = (f"log10(NTU) = {lr.intercept_:.2f} + {lr.coef_[0]:.2f} · ln(decay_len_inner / px), "
                      f"valid for decay_len_inner ≥ {bundle['model'].thr_:.1f} px")

# ---- 9. robustness ---------------------------------------------------------------------------------
r8["abs_dpred"] = r8.dpred_pct.abs()
hi8 = r8[r8.ntu >= 200]
rb = hi8.groupby(["test", "level"]).abs_dpred.agg(["median", "max"]).round(1)
rb["fit band intact (all samples)"] = r8.groupby(["test", "level"]).band_ok.min().map(lambda v: f"{100*v:.0f} %")
rb["class flips (24/6 runs)"] = r8.groupby(["test", "level"]).class_flip.sum()
rb = rb.reset_index()
rb.columns = ["test", "level", "median abs Δpred % (≥200 NTU)", "worst abs Δpred %", "fit band intact (worst)",
              "floor-class flips"]
add("robustness", rb, "Robustness: change in predicted NTU (one shot per sample)")

# ---- 10. parameters ----------------------------------------------------------------------------------
pa = pd.DataFrame([(k, json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in params.items()],
                  columns=["parameter", "value"])
add("params", pa, "Frozen preprocessing parameters (models/params.json)")


# ---- write tables ------------------------------------------------------------------------------------
def to_md(df):
    df = df.astype(str)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in df.values])


def to_png(df, title, path):
    df = df.astype(str)
    widths = np.array([max(len(c), df[c].str.len().max()) for c in df.columns], float) + 2
    w, h = min(0.085 * widths.sum(), 30), 0.3 * (len(df) + 1)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", bbox=[0, 0, 1, 1],
                 colWidths=list(widths / widths.sum()))
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (i, _), cell in t.get_celld().items():
        if i == 0:
            cell.set_facecolor("#1f3b57")
            cell.set_text_props(color="white", weight="bold")
        elif i % 2 == 0:
            cell.set_facecolor("#eef2f6")
    ax.set_title(title, fontsize=11, weight="bold", loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


md = {}
for name, (df, title) in TABLES.items():
    df.to_csv(TAB / f"{name}.csv", index=False)
    md[name] = to_md(df)
    (TAB / f"{name}.md").write_text(f"**{title}**\n\n{md[name]}\n")
    to_png(df, title, TAB / "png" / f"{name}.png")

# ---- fill the template -------------------------------------------------------------------------------
tpl = (ROOT / "docs" / "explanation_template.md").read_text()


def fill(m):
    kind, key = m.group(1), m.group(2)
    if kind == "T":
        df, title = TABLES[key]
        return f"*Table – {title}* (`outputs/tables/{key}.csv`, slide image `outputs/tables/png/{key}.png`)\n\n{md[key]}"
    return str(VALUES[key])


doc = re.sub(r"\{\{(T|V):([A-Za-z0-9_]+)\}\}", fill, tpl)
left = re.findall(r"\{\{[^}]+\}\}", doc)
assert not left, f"unfilled placeholders: {left}"
(ROOT / "explanation.md").write_text(doc)
print(f"{len(TABLES)} tables -> outputs/tables/ (csv, md, png); explanation.md written")
