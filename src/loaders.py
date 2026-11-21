"""DNG loading: keep everything LINEAR, cache to disk so reruns are fast."""
import json
import os
from pathlib import Path

import numpy as np
import rawpy
import exifread

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get("TURBIDITY_DATA_DIR", ROOT / "data" / "calibration_dng"))  # calibration DNGs


def read_meta(path):
    """Sensor + EXIF facts we care about. Bayer vs ProRAW is decided here."""
    path = Path(path)
    with rawpy.imread(str(path)) as r:
        desc = r.color_desc.decode()                 # e.g. 'RGBG'
        pat = r.raw_pattern                          # 2x2 indices into desc
        cfa = "".join(desc[i] for i in pat.flatten())  # e.g. 'BGGR'
        meta = dict(
            file=path.name,
            raw_type=str(r.raw_type).split(".")[-1],  # Flat = Bayer mosaic, Stack = demosaiced (ProRAW-like)
            num_colors=int(r.num_colors),
            cfa=cfa,
            black=float(np.mean(r.black_level_per_channel)),
            white=float(r.white_level),
            height=int(r.raw_image_visible.shape[0]),
            width=int(r.raw_image_visible.shape[1]),
            cam_wb=[float(x) for x in r.camera_whitebalance],
        )
    with open(path, "rb") as fh:
        tags = exifread.process_file(fh, details=False)
    g = lambda k: str(tags.get(k, ""))
    meta.update(
        exposure=g("EXIF ExposureTime"), iso=g("EXIF ISOSpeedRatings"),
        fnumber=g("EXIF FNumber"), focal35=g("EXIF FocalLengthIn35mmFilm"),
        lens=g("EXIF LensModel"), software=g("Image Software"),
        datetime=g("EXIF DateTimeOriginal"),
    )
    meta["bit_depth"] = int(np.ceil(np.log2(meta["white"] + 1)))
    return meta


def load_mosaic(path):
    """Raw Bayer mosaic as uint16 (untouched sensor values). Cached as .npy."""
    path = Path(path)
    npy = CACHE / f"{path.stem}_mosaic.npy"
    if npy.exists():
        return np.load(npy)
    with rawpy.imread(str(path)) as r:
        m = r.raw_image_visible.copy()
    np.save(npy, m)
    return m


def normalize(mosaic, black, white):
    """Black-subtract and scale so 0 = black level, 1 = sensor clipping."""
    return np.clip((mosaic.astype(np.float32) - black) / (white - black), -0.05, None)


def cfa_planes(norm, cfa):
    """Split the mosaic into its 4 half-resolution colour planes with NO demosaic.
    Returns dict like {'R':..., 'G1':..., 'G2':..., 'B':...}."""
    offs = {(0, 0): cfa[0], (0, 1): cfa[1], (1, 0): cfa[2], (1, 1): cfa[3]}
    out, gcount = {}, 0
    for (dy, dx), c in offs.items():
        if c == "G":
            gcount += 1
            c = f"G{gcount}"
        out[c] = norm[dy::2, dx::2]
    return out


def demosaic_linear(path):
    """Full-res linear RGB via LibRaw (AHD demosaic), no gamma, no WB, no auto-brightening.
    Values are in raw DN after LibRaw's own black subtraction; we normalise afterwards."""
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False,
                            user_wb=[1, 1, 1, 1], output_color=rawpy.ColorSpace.raw,
                            output_bps=16, no_auto_scale=True, user_flip=0)
    return rgb


def load_cached_meta(path):
    """Meta dict, cached as JSON next to the mosaic."""
    path = Path(path)
    js = CACHE / f"{path.stem}_meta.json"
    if js.exists():
        return json.loads(js.read_text())
    m = read_meta(path)
    js.write_text(json.dumps(m, indent=1))
    return m
