"""analyse(): DNG -> prediction + QC verdict. The one entry point shared by predict.py and the app."""
from pathlib import Path

import numpy as np

from model import predict_ntu
from pipeline import PARAMS, extract

# QC problems that make the number meaningless -> no NTU is reported
BLOCKING = ("no_spot", "fit_band_saturated", "roi_off_frame")


def analyse(path, bundle):
    o = extract(path, PARAMS, calib_exif=bundle["calib_exif"], extras=False)
    blocking = [q for q in o["qc"] if q.startswith(BLOCKING)]
    warnings = [q for q in o["qc"] if not q.startswith(BLOCKING)]
    res = dict(file=Path(path).name, status="ok", ntu=None, ntu_low=None, ntu_high=None,
               below_detection=None, message="", qc_blocking=blocking, qc_warnings=warnings,
               centre=[o.get("cy"), o.get("cx")], features={})
    if "features" in o:
        res["features"] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                           for k, v in o["features"].items()}
    if blocking:
        res.update(status="rejected", message="not analysed: " + "; ".join(blocking))
        return res, o
    p = predict_ntu(bundle, o["features"])
    if p["below_floor"]:      # the floor class is a category, not a measurement: report only a bound
        res.update(below_detection=True, ntu_upper_bound=round(bundle["detection_floor_ntu"]), message=p["message"])
    else:
        res.update(ntu=round(p["ntu"], 1), ntu_low=round(p["low"], 1), ntu_high=round(p["high"], 1),
                   below_detection=False, message=p["message"])
    if warnings or p["extrapolating"]:
        res["status"] = "ok_with_warnings"
    return res, o
