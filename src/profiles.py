"""Radial profiles of the laser disc. NaN-aware, saturation-aware, on linear CFA planes."""
import numpy as np


def radial_profile(roi, sat_mask=None, bin_px=5, r_max=None, max_sat_frac=0.01):
    """Azimuthal mean intensity in rings of width `bin_px` around the ROI centre.

    roi       : 2-D float plane, ROI centred on the spot (NaN = outside frame).
    sat_mask  : same-shape bool, True where the pixel is (near) clipped.
                A ring with more than `max_sat_frac` saturated pixels is set to NaN:
                averaging only the unclipped pixels of a ring would bias it low.
    Returns r (ring centres), mean, std, n_valid (all 1-D, same length)."""
    h, w = roi.shape
    yy, xx = np.indices(roi.shape)
    rad = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    r_max = r_max if r_max is not None else min(h, w) / 2
    edges = np.arange(0, r_max + bin_px, bin_px)
    idx = np.digitize(rad.ravel(), edges) - 1
    vals = roi.ravel()
    sat = sat_mask.ravel() if sat_mask is not None else np.zeros(vals.size, bool)
    nb = len(edges) - 1
    keep = (idx >= 0) & (idx < nb) & ~np.isnan(vals)
    n_all = np.bincount(idx[keep], minlength=nb)
    n_sat = np.bincount(idx[keep], weights=sat[keep], minlength=nb)
    good = keep & ~sat
    n = np.bincount(idx[good], minlength=nb).astype(float)
    s1 = np.bincount(idx[good], weights=vals[good], minlength=nb)
    s2 = np.bincount(idx[good], weights=vals[good] ** 2, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
        bad = (n == 0) | (n_sat / np.maximum(n_all, 1) > max_sat_frac)
    mean[bad], std[bad] = np.nan, np.nan
    return 0.5 * (edges[:-1] + edges[1:]), mean, std, n.astype(int)
