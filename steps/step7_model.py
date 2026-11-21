"""Step 7: models + leave-one-sample-out CV. Final model is fitted on all data and saved to models/.

Effective sample size is 6 (one cuvette per NTU level). Leave-one-group-out by sample_id holds out a
whole NTU level per fold, so 0 and 800 NTU are always pure extrapolation.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loaders import DATA_DIR, ROOT
from pipeline import PARAMS, EXIF_KEYS, extract, save_params
from model import FloorLogLog, to_y, to_ntu

OUT, MODELS = ROOT / "outputs", ROOT / "models"
MODELS.mkdir(exist_ok=True)
labels = pd.read_csv(ROOT / "labels.csv")

# ---- 1. features through the SAME pipeline the app uses; must match the step 4/5 scripts --------
rows, exifs = [], []
for _, lab in labels.iterrows():
    o = extract(DATA_DIR / lab.file)
    assert not o["qc"], f"{lab.file}: {o['qc']}"
    rows.append(dict(file=lab.file, ntu=lab.ntu, sample_id=lab.sample_id, shot=lab.shot, **o["features"]))
    exifs.append({k: o["meta"][k] for k in EXIF_KEYS})
D = pd.DataFrame(rows)
f4 = pd.read_csv(OUT / "step4_features.csv").set_index("file")
err = np.abs(D.set_index("file").decay_len_inner - f4.decay_len_inner).max()
assert err < 1e-6, f"pipeline differs from step 4 by {err}"
print(f"pipeline vs step-4 decay_len_inner: max abs diff {err:.1e} (OK)")
calib_exif = exifs[0]
assert all(e == calib_exif for e in exifs), "EXIF differs within calibration set"
D.to_csv(OUT / "step7_pipeline_features.csv", index=False)

log_tf = FunctionTransformer(np.log)
MODELS_CV = {
    "null (train mean)": (DummyRegressor(), ["decay_len_inner"]),
    "log-log, 1 feature": (make_pipeline(log_tf, LinearRegression()), ["decay_len_inner"]),
    "log-log >=200 + floor": (FloorLogLog(), ["decay_len_inner"]),
    "ridge, 2 features": (make_pipeline(log_tf, StandardScaler(), Ridge(alpha=1.0)),
                          ["decay_len_inner", "core_halo_ratio"]),
    "GBM, 2 features": (GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1,
                                                  random_state=0), ["decay_len_inner", "core_halo_ratio"]),
}

# ---- 2. leave-one-sample-out CV ---------------------------------------------------------------
y, groups = to_y(D.ntu.values), D.sample_id.values
logo = LeaveOneGroupOut()
pred = {}
for name, (est, cols) in MODELS_CV.items():
    p = np.empty(len(D))
    for tr, te in logo.split(D, y, groups):
        m = clone(est).fit(D.iloc[tr][cols].values, y[tr])     # scaler/floor fitted inside the fold
        p[te] = m.predict(D.iloc[te][cols].values)
    pred[name] = p


def metrics(ntu, p_y):
    p = to_ntu(p_y)
    t_y = to_y(ntu)
    ss = ((ntu - ntu.mean()) ** 2).sum()
    return dict(MAE=np.mean(abs(p - ntu)), RMSE=np.sqrt(np.mean((p - ntu) ** 2)),
                R2=1 - ((p - ntu) ** 2).sum() / ss, RMSE_log=np.sqrt(np.mean((p_y - t_y) ** 2)))


pd.set_option("display.width", 220)
print("\n== LOGO CV (6 folds = 6 samples). NTU-space metrics + RMSE of log10(NTU+1) ==")
M = pd.DataFrame({k: metrics(D.ntu.values, v) for k, v in pred.items()}).T
print(M.round(3).to_string())
print("\n== per-level: mean LOGO prediction (NTU) / mean abs error ==")
lv = pd.DataFrame({k: D.assign(p=to_ntu(v)).groupby("ntu").p.mean() for k, v in pred.items()}).round(0)
print(lv.T.to_string())
M.rename_axis("model").to_csv(OUT / "step7_metrics.csv")
lv.T.rename_axis("model").to_csv(OUT / "step7_per_level.csv")

# ---- 3. final model: best LOGO RMSE_log among real models, refit on all data ---------------------
best = M.drop(index="null (train mean)").RMSE_log.idxmin()
est, cols = MODELS_CV[best]
final = clone(est).fit(D[cols].values, y)
# Uncertainty: quoted only for predictions above the detection floor, from held-out residuals of the
# quantifiable levels in log10(NTU). (log10(NTU+1) residuals are dominated by 0 vs ~20 NTU, which
# says nothing about the error at 400 NTU.)
q = D.ntu.values >= 200
resid_sd = float(np.sqrt(np.mean((np.log10(to_ntu(pred[best][q]) + 1e-9) - np.log10(D.ntu.values[q])) ** 2)))
floor_ntu = (float(to_ntu(final.lr_.predict(np.log([[final.thr_]])))[0]) if hasattr(final, "thr_")
             else 50.0)
low_ok = (to_ntu(pred[best][~q]) < floor_ntu).mean()
hi_ok = (to_ntu(pred[best][q]) >= floor_ntu).mean()
fit_in = final.predict(D[cols].values)
bundle = dict(
    model=final, name=best, features=cols, target="log10(NTU+1)",
    logo_rmse_log10_above_floor=resid_sd, logo_low_class_acc=float(low_ok), logo_high_class_acc=float(hi_ok), logo_metrics=M.loc[best].to_dict(),
    # Lowest NTU the model can output above its threshold. Everything below is one class: 0/10/50 NTU
    # were indistinguishable (steps 4-6) and 50-200 NTU was never measured, so this is the honest floor.
    detection_floor_ntu=floor_ntu,
    train_feature_range={c: [float(D[c].min()), float(D[c].max())] for c in cols},
    train_ntu_levels=sorted(int(v) for v in D.ntu.unique()),
    calib_exif=calib_exif, n_independent_samples=int(D.sample_id.nunique()),
)
joblib.dump(bundle, MODELS / "turbidity_model.joblib")
save_params(MODELS / "params.json", PARAMS)
pd.DataFrame(dict(file=D.file, ntu=D.ntu, in_sample=to_ntu(fit_in), logo=to_ntu(pred[best]))).to_csv(
    OUT / "step7_predictions.csv", index=False)
print(f"\nfinal model: {best}; features {cols}")
if hasattr(final, "named_steps") and "linearregression" in final.named_steps:
    lr = final.named_steps["linearregression"]
    print(f"  log10(NTU+1) = {lr.intercept_:.3f} + {lr.coef_[0]:.3f} * ln(decay_len_inner)")
print(f"  quantifiable floor: {floor_ntu:.0f} NTU (regression value at the learned threshold)")
print(f"  held-out 0/10/50 NTU images called 'below floor': {100*low_ok:.0f}%; "
      f"held-out 200/400/800 called quantifiable: {100*hi_ok:.0f}%")
print(f"  held-out error for 200-800 NTU: RMSE {resid_sd:.3f} log10 units -> x/÷ {10**resid_sd:.2f} (1 sd), "
      f"x/÷ {10**(2*resid_sd):.2f} (2 sd)")
print(f"  in-sample fit (all data) at 200/400/800: " + ", ".join(
    f"{n}->{to_ntu(fit_in)[D.ntu.values == n].mean():.0f}" for n in (200, 400, 800)))
print("saved: models/turbidity_model.joblib, models/params.json, outputs/step7_predictions.csv")

# ---- 4. plots ---------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
ax = axes[0]
x = D.decay_len_inner
ax.scatter(np.log(x), y, c=np.log1p(D.ntu), cmap="viridis", s=25)
xx = np.linspace(np.log(x.min()) - 0.05, np.log(x.max()) + 0.05, 100)
ax.plot(xx, final.predict(np.exp(xx)[:, None]), "k-", lw=2, label=f"final: {best}")
for fold_sid in D.sample_id.unique():                     # LOGO lines show how unstable the fit is
    tr = D.sample_id != fold_sid
    m = clone(MODELS_CV["log-log, 1 feature"][0]).fit(D[tr][["decay_len_inner"]].values, y[tr])
    ax.plot(xx, m.predict(np.exp(xx)[:, None]), "-", color="gray", lw=0.6, alpha=0.7)
ax.set_xlabel("ln(decay_len_inner / px)")
ax.set_ylabel("log10(NTU + 1)")
ax.set_title("black = final model; grey = 1-feature log-log LOGO refits")
ax.legend()
ax = axes[1]
lim = [0, 1000]
for k, name in enumerate([n for n in pred if not n.startswith("null")]):
    ax.scatter(D.ntu + (k - 1.5) * 12, to_ntu(pred[name]), s=14, label=name, alpha=0.8)
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(-50, 900)
ax.set_ylim(-50, max(1000, ax.get_ylim()[1]))
ax.set_xlabel("true NTU")
ax.set_ylabel("LOGO-predicted NTU (held-out sample)")
ax.set_title("held-out predictions (x jittered per model)")
ax.legend(fontsize=8)
ax = axes[2]
names = [n for n in pred]
ax.barh(names, M.RMSE_log, color=["gray"] + ["C0"] * (len(names) - 1))
ax.set_xlabel("LOGO RMSE of log10(NTU+1)  (lower = better)")
ax.set_title(f"final: {best}")
fig.suptitle("Step 7: models with leave-one-sample-out CV (6 independent samples)")
fig.tight_layout()
fig.savefig(OUT / "step7_model.png", dpi=95)
