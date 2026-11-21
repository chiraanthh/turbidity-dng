"""Step 6: every candidate feature vs NTU side by side -> Spearman, repeat error bars, separation,
redundancy -> a short, defensible shortlist (final pick is made with the user)."""
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
from loaders import ROOT, CACHE

OUT = ROOT / "outputs"
f4 = pd.read_csv(OUT / "step4_features.csv")
f5 = pd.read_csv(OUT / "step5_fft_features.csv")
F = f4.merge(f5, on=["file", "ntu", "sample_id", "shot"], validate="one_to_one")

# candidate -> (source step, known confound or None). Confounded features are shown but never shortlisted.
CANDIDATES = {
    "decay_len_inner":  ("4 profile", None),
    "decay_len_halo":   ("4 profile", None),
    "core_halo_ratio":  ("4 profile", None),
    "max_grad":         ("4 profile", "steepest point sits on the clipped-core edge for 0-50 NTU"),
    "lap_var":          ("4 image", None),
    "tenengrad":        ("4 image", None),
    "spec_slope_cclip": ("5 fft", None),
    "hf_ratio_asis":    ("5 fft", "tracks the clipped-plateau edge (trend vanishes under common clip)"),
    "f50_cclip":        ("5 fft", "tracks plateau size above T -> core brightness"),
    "f_noise_asis":     ("5 fft", "signal-to-noise: dimmer image hits noise floor sooner"),
}
NTUS = sorted(F.ntu.unique())
SIGNAL_PAIRS = ["sep 50-200", "sep 200-400", "sep 400-800"]   # the range where any feature carries signal
REDUNDANT = 0.95     # |Pearson r| on sample means above which two features count as the same measurement

# ---- 1. per-feature statistics ----------------------------------------------------------------
g = F.groupby("ntu")
rows = []
for c, (src, confound) in CANDIDATES.items():
    mu, sd = g[c].mean(), g[c].std()
    rho6, p6 = spearmanr(NTUS, mu.values)
    rho30 = spearmanr(F.ntu, F[c], nan_policy="omit")[0]
    # adjacent-level separation in units of shot sd: |Δmean| / sqrt(sd1² + sd2²)  (optimistic: shot sd
    # is within ONE cuvette, not between independent samples of the same NTU)
    with np.errstate(divide="ignore"):
        sep = [abs(mu[b] - mu[a]) / np.hypot(sd[a], sd[b]) for a, b in zip(NTUS[:-1], NTUS[1:])]
    # does the sample-mean order match NTU order? count adjacent pairs in the right direction
    sign = np.sign(rho6)
    right = sum(sign * (mu[b] - mu[a]) > 0 for a, b in zip(NTUS[:-1], NTUS[1:]))
    rows.append(dict(feature=c, step=src, rho6=rho6, p6=p6, rho30=rho30, ordered_pairs=f"{right}/5",
                     **{f"sep {a}-{b}": s for (a, b), s in zip(zip(NTUS[:-1], NTUS[1:]), sep)},
                     cv_shots=float((sd / mu.abs()).mean()), confound=confound or ""))
S = pd.DataFrame(rows).set_index("feature")
pd.set_option("display.width", 250)
print("== candidates. sep a-b = |Δmean| / combined shot sd (>3 ~ resolvable, but optimistic); "
      "cv_shots = mean shot sd / |mean| ==")
print(S.drop(columns="confound").round(2).to_string())
print("\nconfounded (shown, not shortlisted):")
for c, r in S[S.confound != ""].iterrows():
    print(f"  {c:16s} {r.confound}")

# ---- 2. redundancy (Pearson r on the 6 sample means) ------------------------------------------
M = g[list(CANDIDATES)].mean()
C = M.corr()
clean = S[S.confound == ""].copy()
# Ranking: rho6 can't separate features (one shared misordering at 10 NTU gives most of them 0.83),
# so rank by the WORST adjacent separation over 50->200->400->800, the range where signal exists.
clean["score"] = clean[SIGNAL_PAIRS].min(axis=1)
order = clean.sort_values("score", ascending=False).index
picked = []
for c in order:
    if all(abs(C.loc[c, p]) < REDUNDANT for p in picked):
        picked.append(c)
# cross-checks: best clean feature from each OTHER computation family (different failure modes)
fam = lambda c: CANDIDATES[c][0]
backups = [next(c for c in order if fam(c) == f) for f in dict.fromkeys(fam(c) for c in order) if f != fam(picked[0])]
print("\n== ranking by worst separation over 50->800 ==")
print(clean.loc[order, SIGNAL_PAIRS + ["score", "rho6", "cv_shots"]].round(2).to_string())
print(f"\n== Pearson r between sample means (clean features) ==")
print(C.loc[clean.index, clean.index].round(3).to_string())
print(f"\nshortlist (non-redundant at |r| < {REDUNDANT}): {picked}")
print(f"cross-check features (one per other family, redundant by construction): {backups}")

# ---- 3. plots ---------------------------------------------------------------------------------
feats = list(CANDIDATES)
fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(3, 6)
for k, c in enumerate(feats):
    ax = fig.add_subplot(gs[k // 4, k % 4])
    z_mu, z_sd = M[c], g[c].std()
    ax.errorbar(range(len(NTUS)), z_mu, z_sd, fmt="o-", capsize=4, color="C3" if CANDIDATES[c][1] else "C0")
    for i, n in enumerate(NTUS):
        v = F[F.ntu == n][c]
        ax.plot(i + np.linspace(-0.12, 0.12, len(v)), v, ".", color="gray", alpha=0.6)
    ax.set_xticks(range(len(NTUS)), [str(n) for n in NTUS], fontsize=8)
    tag = " [confound]" if CANDIDATES[c][1] else (" [picked]" if c in picked else " [cross-check]" if c in backups else "")
    ax.set_title(f"{c}{tag}\nrho6={S.loc[c, 'rho6']:+.2f}  ordered {S.loc[c, 'ordered_pairs']}", fontsize=9)
    ax.grid(alpha=0.3)
ax = fig.add_subplot(gs[:2, 4:])
cc = C.loc[clean.index, clean.index]
im = ax.imshow(cc.abs(), vmin=0.5, vmax=1, cmap="magma")
ax.set_xticks(range(len(cc)), cc.index, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(cc)), cc.index, fontsize=8)
for i in range(len(cc)):
    for j in range(len(cc)):
        ax.text(j, i, f"{cc.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                color="k" if abs(cc.iloc[i, j]) > 0.8 else "w")
ax.set_title("|Pearson r| between sample means (clean features)")
fig.colorbar(im, ax=ax, fraction=0.04)
fig.suptitle("Step 6: feature vs NTU (x categorical). Error bars = sd over 5 burst shots (one cuvette per level). "
             "Red = known confound")
fig.tight_layout()
fig.savefig(OUT / "step6_feature_selection.png", dpi=95)

S.to_csv(OUT / "step6_feature_stats.csv")
(CACHE / "step6_selection.json").write_text(json.dumps(dict(
    shortlist=picked, cross_checks=backups, ranking="min separation over " + ", ".join(SIGNAL_PAIRS),
    redundancy_threshold=REDUNDANT,
    excluded_confounded={c: v[1] for c, v in CANDIDATES.items() if v[1]}), indent=1))
print("\nsaved: outputs/step6_feature_selection.png, outputs/step6_feature_stats.csv, outputs/cache/step6_selection.json")
