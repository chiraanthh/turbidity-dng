"""Single feature path used by training (step 7), robustness tests (step 8), the CLI and the app.

DNG -> B CFA plane (linear, black-subtracted, no demosaic) -> spot centre -> fixed ROI
    -> near-clip mask -> radial profile -> features + QC flags.

Every number that affects the features lives in PARAMS and is saved next to the model, so a
model can't silently be used with different preprocessing.
"""
import json
from pathlib import Path

import numpy as np
import rawpy

from loaders import read_meta
from preprocess import find_centroid, crop_roi
from profiles import radial_profile
from features import profile_features, image_features, fft_features, INNER_BAND

PARAMS = dict(
    version=1,
    channel="B",                 # least-clipped plane: red laser leaks weakly into B pixels
    centroid=dict(sigma=10, frac=0.5, floor_pct=5),
    half=600,                    # ROI half-size (plane px), step 2: worst-case off-frame <= 1%
    r_full=595,                  # profile radius fully inside the frame for every calibration image
    bin_px=5,                    # ring width (plane px)
    near_clip_dn=64,             # pixels within 64 DN of white count as saturated
    max_nan=0.01,                # QC: max ROI fraction off-frame
    common_clip=0.464,           # step 5: shared clip level for spec_slope_cclip
    inner_band=list(INNER_BAND),
    min_edge_px=450,             # QC: spot centre must be this far from the plane edge
)

EXIF_KEYS = ["iso", "exposure", "fnumber", "focal35", "lens"]


def load_b_plane(path):
    """Return (B plane normalised so 0 = black, 1 = white level; meta dict incl. EXIF)."""
    meta = read_meta(path)
    with rawpy.imread(str(path)) as r:
        mos = r.raw_image_visible.astype(np.float32)
    if meta["raw_type"] != "Flat" or meta["num_colors"] != 3:
        raise ValueError(f"{Path(path).name}: not a Bayer RAW (raw_type={meta['raw_type']}); ProRAW is unsupported")
    offs = {meta["cfa"][0]: (0, 0), meta["cfa"][3]: (1, 1)}       # BGGR / RGGB: B is on the diagonal
    if "B" not in offs:
        raise ValueError(f"unsupported CFA {meta['cfa']}")
    dy, dx = offs["B"]
    B = (mos[dy::2, dx::2] - meta["black"]) / (meta["white"] - meta["black"])
    return B, meta


def sat_threshold(meta, params=PARAMS):
    """Normalised value equivalent to raw DN >= white - near_clip_dn (half-DN guard against float rounding)."""
    return (meta["white"] - params["near_clip_dn"] - 0.5 - meta["black"]) / (meta["white"] - meta["black"])


def features_from_plane(B, cy, cx, sat_level, params=PARAMS, extra_sat=None, extras=True):
    """Features for a B plane with the spot centre given. `extra_sat` (same shape as B) is OR-ed into
    the saturation mask; robustness tests use it to keep pixels that were clipped before rescaling."""
    roi = crop_roi(B, cy, cx, params["half"])
    with np.errstate(invalid="ignore"):
        sat = roi >= sat_level
    if extra_sat is not None:
        sat |= crop_roi(extra_sat.astype(np.float32), cy, cx, params["half"]) > 0.5
    r, I, _, _ = radial_profile(roi, sat, params["bin_px"], r_max=params["r_full"])
    f = profile_features(r, I)
    if extras:
        f["tenengrad"] = image_features(roi, sat)["tenengrad"]
        f["spec_slope_cclip"] = fft_features(roi, params["common_clip"])["spec_slope"]
    band = (r >= params["inner_band"][0]) & (r < params["inner_band"][1])
    f["fit_band_valid_frac"] = float(np.isfinite(I[band]).mean())
    f["roi_nan_frac"] = float(np.isnan(roi).mean())
    return f, r, I


def extract(path, params=PARAMS, calib_exif=None, extras=True):
    """DNG path -> dict with centre, features, profile and a list of QC problems (empty = clean)."""
    B, meta = load_b_plane(path)
    out = dict(file=Path(path).name, meta=meta, qc=[])
    c = find_centroid(B, **params["centroid"])
    if c is None:
        out["qc"].append("no_spot: nothing stands out above background")
        return out
    out["cy"], out["cx"] = c
    f, r, I = features_from_plane(B, *c, sat_threshold(meta, params), params, extras=extras)
    out.update(features=f, r=r, profile=I)
    H, W = B.shape
    edge = min(c[0], c[1], H - c[0], W - c[1])
    if edge < params["min_edge_px"]:
        out["qc"].append(f"near_edge: spot centre {edge:.0f} px from frame edge (< {params['min_edge_px']})")
    if f["roi_nan_frac"] > params["max_nan"]:
        out["qc"].append(f"roi_off_frame: {100*f['roi_nan_frac']:.1f}% of ROI outside frame")
    if f["fit_band_valid_frac"] < 1:
        out["qc"].append(f"fit_band_saturated: {100*(1-f['fit_band_valid_frac']):.0f}% of rings in "
                         f"{params['inner_band']} px are clipped -> shorten exposure")
    if calib_exif:
        diff = {k: (meta.get(k), v) for k, v in calib_exif.items() if str(meta.get(k)) != str(v)}
        if diff:
            out["qc"].append("exif_mismatch: " + ", ".join(f"{k}={a} (calib {b})" for k, (a, b) in diff.items()))
    return out


def save_params(path, params=PARAMS):
    Path(path).write_text(json.dumps(params, indent=1))


def check_params(saved):
    """Refuse to run if the model was trained with different preprocessing."""
    diff = {k for k in set(PARAMS) | set(saved) if PARAMS.get(k) != saved.get(k)}
    if diff:
        raise RuntimeError(f"model params differ from code params: {sorted(diff)}; retrain (steps/step7_model.py)")
