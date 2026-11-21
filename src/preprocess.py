"""Preprocessing: laser spot centre, fixed ROI crop, saturation checks. All on linear CFA planes."""
import numpy as np
from scipy.ndimage import gaussian_filter

CHANNELS = ["R", "G1", "G2", "B"]


def find_centroid(plane, sigma=10, frac=0.5, floor_pct=5):
    """Intensity-weighted centre of the spot.
    Smooth (kills pixel noise), keep pixels above `frac` of the peak, weight by intensity.
    Background floor = low percentile, NOT the median: the disc can cover >50% of the frame,
    so the median would be disc light, not background.
    Returns (cy, cx) in plane pixels, or None if no clear spot."""
    s = gaussian_filter(np.nan_to_num(plane), sigma)
    peak, floor = s.max(), np.percentile(s, floor_pct)
    if peak - floor < 0.02:                      # nothing stands out above the background
        return None
    w = np.where(s >= floor + frac * (peak - floor), s - floor, 0.0)
    yy, xx = np.indices(s.shape)
    return float((w * yy).sum() / w.sum()), float((w * xx).sum() / w.sum())


def crop_roi(plane, cy, cx, half):
    """Square (2*half) ROI centred on (cy, cx). Parts outside the frame become NaN,
    so later radial averages simply ignore them instead of seeing fake zeros."""
    out = np.full((2 * half, 2 * half), np.nan, dtype=np.float32)
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + 2 * half, plane.shape[0]), min(x0 + 2 * half, plane.shape[1])
    out[ys - y0:ye - y0, xs - x0:xe - x0] = plane[ys:ye, xs:xe]
    return out


def clip_fraction(roi):
    """Fraction of valid pixels at or above sensor clip (normalised value >= 1.0, i.e. raw DN >= white level)."""
    v = roi[~np.isnan(roi)]
    return float((v >= 1.0).mean())


def raw_planes(mosaic, cfa):
    """Same split as loaders.cfa_planes but on untouched uint16 DN (for clip checks against white)."""
    offs = {(0, 0): cfa[0], (0, 1): cfa[1], (1, 0): cfa[2], (1, 1): cfa[3]}
    out, g = {}, 0
    for (dy, dx), c in offs.items():
        if c == "G":
            g += 1
            c = f"G{g}"
        out[c] = mosaic[dy::2, dx::2]
    return out
