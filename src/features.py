"""Blur/scatter features from a radial profile and from the 2-D ROI.

Two families:
  * profile features (1-D): computed from the azimuthal-mean profile; features that need the true
    peak return NaN when the core is clipped, rather than a silently wrong number.
  * image features (2-D): Laplacian variance and Tenengrad, inside a fixed annulus, near-clipped
    pixels excluded, normalised by mean intensity squared so they measure shape, not brightness.
"""
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_laplace, sobel
from scipy.signal import savgol_filter

INNER_BAND = (300, 450)    # flank: valid (unclipped) in every image, B channel
HALO_BAND = (450, 590)     # outer halo, inside r_full
CORE_R, HALO_R = 300, 500  # core-to-halo intensity ratio radii
ANNULUS = (300, 590)       # 2-D features region


def _log_slope(r, I, band):
    k = (r >= band[0]) & (r < band[1]) & np.isfinite(I) & (I > 0)
    if k.sum() < 5:
        return np.nan
    return float(np.polyfit(r[k], np.log(I[k]), 1)[0])


def _at(r, I, r0):
    return float(I[np.argmin(np.abs(r - r0))])


def _radius_where(r, I, level):
    """Smallest radius after the peak where the profile falls below `level` (linear interp)."""
    below = np.nonzero(I < level)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    return float(np.interp(level, [I[i], I[i - 1]], [r[i], r[i - 1]]))


def profile_features(r, I):
    """r, I: ring centres and azimuthal means (NaN = saturated / missing ring)."""
    finite = np.isfinite(I)
    r_first = float(r[np.argmax(finite)])
    peak_ok = bool(finite[0])                  # centre ring unclipped -> true peak is measured
    f = dict(r_first_valid=r_first, peak_ok=peak_ok)

    s_in, s_halo = _log_slope(r, I, INNER_BAND), _log_slope(r, I, HALO_BAND)
    f["slope_inner"] = s_in                    # d lnI / dr on the flank (edge sharpness, 1/px)
    f["slope_halo"] = s_halo
    f["decay_len_inner"] = -1 / s_in if s_in < 0 else np.nan   # px for intensity to drop by e
    f["decay_len_halo"] = -1 / s_halo if s_halo < 0 else np.nan
    f["core_halo_ratio"] = _at(r, I, CORE_R) / _at(r, I, HALO_R)

    # inflection radius: d2I/dr2 changes sign from - (shoulder) to + (tail), on a smoothed profile
    f["inflection_r"] = np.nan
    seg = np.nonzero(finite)[0]
    if len(seg) > 21:
        Is = savgol_filter(I[seg], 21, 3)
        d2 = savgol_filter(I[seg], 21, 3, deriv=2)
        flips = np.nonzero((d2[:-1] < 0) & (d2[1:] >= 0))[0]
        if len(flips) and peak_ok:             # without the true shoulder an inflection is meaningless
            f["inflection_r"] = float(r[seg][flips[0]])
        f["max_grad"] = float(-np.min(np.gradient(Is, r[seg])))    # steepest drop, 1/px
    else:
        f["max_grad"] = np.nan

    # peak-referenced widths: only meaningful when the peak is not clipped
    f["fwhm"] = f["w10_90"] = f["peak_halo_ratio"] = np.nan
    if peak_ok:
        pk = np.nanmax(I[:3])
        f["fwhm"] = 2 * _radius_where(r, I, 0.5 * pk)
        f["w10_90"] = _radius_where(r, I, 0.1 * pk) - _radius_where(r, I, 0.9 * pk)
        halo = (r >= HALO_BAND[0]) & (r < HALO_BAND[1])
        f["peak_halo_ratio"] = float(pk / np.nanmean(I[halo]))
    return f


def image_features(roi, sat_mask, annulus=ANNULUS, sigma=2.0):
    """Laplacian variance and Tenengrad in an annulus, brightness-normalised.
    sigma: pre-smoothing (plane px) so pixel noise doesn't dominate the 2nd derivative."""
    h, w = roi.shape
    yy, xx = np.indices(roi.shape)
    rad = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    bad = np.isnan(roi) | binary_dilation(sat_mask, iterations=int(3 * sigma) + 1)
    img = np.where(np.isnan(roi), 0, roi)
    lap = gaussian_laplace(img, sigma)
    sm = gaussian_filter(img, sigma)
    ten = sobel(sm, 0) ** 2 + sobel(sm, 1) ** 2
    k = (rad >= annulus[0]) & (rad < annulus[1]) & ~bad
    m2 = float(np.mean(roi[k])) ** 2
    return dict(lap_var=float(np.var(lap[k]) / m2), tenengrad=float(np.mean(ten[k]) / m2),
                annulus_valid_frac=float(k.sum() / ((rad >= annulus[0]) & (rad < annulus[1])).sum()))


# ---------------------------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------------------------
# The blur's spectrum falls into the sensor-noise floor at ~0.008 cycles/px (period ~125 px): the
# disc is nearly as large as the ROI, so only the lowest ~10 FFT bins carry optical signal.
# All signal bands therefore live below 0.008; the noise floor is measured where the spectrum is flat.
PAD = 2                       # zero-padding factor: finer frequency sampling for f50 (not more resolution)
NOISE_BAND = (0.05, 0.15)     # cycles/px: flat white-noise plateau (it rolls off above ~0.2)
LF_BAND = (0.0, 0.003)        # periods > 330 px (DC excluded)
HF_BAND = (0.003, 0.008)      # periods 330..125 px: the finest scale that still rises above noise
SLOPE_BAND = (0.002, 0.007)


def radial_power_spectrum(roi, clip_level=None, pad=PAD):
    """Hann-windowed, zero-padded 2-D power spectrum, averaged/summed over frequency rings.
    clip_level: if set, the ROI is clipped at this level first (same level for every image),
                so the sensor's own clipping (which differs between images) becomes irrelevant.
    Returns f (cycles/px), ring-mean power, ring pixel counts; power normalised by total non-DC power."""
    a = np.where(np.isnan(roi), 0.0, roi).astype(np.float64)
    if clip_level is not None:
        a = np.minimum(a, clip_level)
    a = a * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    n = pad * max(a.shape)
    P = np.abs(np.fft.fftshift(np.fft.fft2(a, s=(n, n)))) ** 2
    yy, xx = np.indices(P.shape)
    k = np.hypot(yy - n // 2, xx - n // 2).astype(int)        # ring index = frequency bin
    nb = n // 2
    ring_sum = np.bincount(k.ravel(), P.ravel(), minlength=nb + 1)[:nb]
    ring_n = np.bincount(k.ravel(), minlength=nb + 1)[:nb]
    return np.arange(nb) / n, ring_sum / ring_n / ring_sum[1:].sum(), ring_n


def fft_features(roi, clip_level=None):
    f, p, ring_n = radial_power_spectrum(roi, clip_level)
    noise = float(np.median(p[(f >= NOISE_BAND[0]) & (f < NOISE_BAND[1])]))
    sig = np.clip(p - noise, 0, None) * ring_n              # noise-free power per ring
    sig[0] = 0                                              # drop DC
    band = lambda b: sig[(f > b[0]) & (f <= b[1])].sum()
    k = (f >= SLOPE_BAND[0]) & (f < SLOPE_BAND[1]) & (p > 3 * noise)
    slope = float(np.polyfit(np.log(f[k]), np.log(p[k] - noise), 1)[0]) if k.sum() > 5 else np.nan
    cum = np.cumsum(sig) / sig.sum()
    above = np.nonzero(p < 3 * noise)[0]
    return dict(hf_ratio=float(band(HF_BAND) / band(LF_BAND)),
                spec_slope=slope,
                f50=float(np.interp(0.5, cum, f)),          # half the noise-free power lies below f50
                f_noise=float(f[above[above > 0][0]]),      # where structure sinks into noise (SNR-dependent!)
                noise_floor=noise)
