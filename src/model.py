"""Target transform and the custom floor model (kept importable so joblib can unpickle it)."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


def to_y(ntu):
    """Regression target. +1 keeps 0 NTU finite; errors become roughly multiplicative."""
    return np.log10(np.asarray(ntu, float) + 1)


def to_ntu(y):
    return np.clip(10 ** np.asarray(y, float) - 1, 0, None)


class FloorLogLog(BaseEstimator, RegressorMixin):
    """log10(NTU) ~ ln(feature) fitted only on levels >= `min_ntu`; anything on the low side of a
    feature threshold is predicted as the mean of the low class. The threshold is learned from the
    training fold (midpoint between the classes), so CV doesn't leak."""

    def __init__(self, min_ntu=200):
        self.min_ntu = min_ntu

    def fit(self, X, y):
        X, ntu = np.asarray(X, float)[:, 0], 10 ** np.asarray(y) - 1
        hi = ntu >= self.min_ntu - 0.5
        self.lr_ = LinearRegression().fit(np.log(X[hi])[:, None], y[hi])
        self.sign_ = np.sign(self.lr_.coef_[0])          # +1: feature grows with NTU
        lo_edge = (X[~hi] * self.sign_).max() if (~hi).any() else -np.inf
        self.thr_ = 0.5 * (lo_edge + (X[hi] * self.sign_).min())
        self.low_y_ = float(np.mean(y[~hi])) if (~hi).any() else float(y[hi].min())
        return self

    def predict(self, X):
        X = np.asarray(X, float)[:, 0]
        p = self.lr_.predict(np.log(X)[:, None])
        return np.where(X * self.sign_ < self.thr_, self.low_y_, p)


def load_bundle(models_dir):
    """Load the trained model and refuse to run if preprocessing params changed since training."""
    import json
    from pathlib import Path
    import joblib
    from pipeline import check_params
    models_dir = Path(models_dir)
    bundle = joblib.load(models_dir / "turbidity_model.joblib")
    check_params(json.loads((models_dir / "params.json").read_text()))
    return bundle


def predict_ntu(bundle, features):
    """features: dict from pipeline. Returns NTU estimate, a 1-sd range and interpretation flags."""
    x = np.array([[features[c] for c in bundle["features"]]], float)
    if not np.all(np.isfinite(x)):
        return dict(ntu=np.nan, low=np.nan, high=np.nan, below_floor=False, extrapolating=False,
                    message="feature could not be computed (see QC)")
    ntu = float(to_ntu(bundle["model"].predict(x))[0])
    fac = 10 ** bundle["logo_rmse_log10_above_floor"]
    lo_hi = bundle["train_feature_range"][bundle["features"][0]]
    extrap = not (lo_hi[0] <= x[0, 0] <= lo_hi[1])
    floor = bundle["detection_floor_ntu"]
    below = ntu < floor
    if below:
        msg = (f"below the quantifiable range (< ~{floor:.0f} NTU). Calibration only resolves 200-800 NTU; "
               f"0-50 NTU look identical and 50-200 NTU was never measured")
    else:
        msg = f"{ntu:.0f} NTU (1-sd range {ntu / fac:.0f}-{ntu * fac:.0f})"
    if extrap:
        msg += f"; WARNING feature outside calibration range {lo_hi} -> extrapolating"
    return dict(ntu=ntu, low=ntu / fac, high=ntu * fac, below_floor=bool(below), extrapolating=bool(extrap),
                message=msg)
