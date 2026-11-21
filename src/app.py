"""Streamlit front-end:  streamlit run src/app.py   (from the project root)."""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from model import load_bundle            # noqa: E402
from pipeline import PARAMS, load_b_plane  # noqa: E402
from service import analyse              # noqa: E402

st.set_page_config(page_title="Laser turbidity", layout="wide")


@st.cache_resource
def bundle():
    return load_bundle(ROOT / "models")


@st.cache_data
def calibration_profiles():
    """Mean reference profile per NTU level (step 3 cache), if present."""
    p = ROOT / "outputs" / "cache" / "step3_profiles.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    with warnings.catch_warnings():                  # clipped rings are NaN in every shot -> empty mean
        warnings.simplefilter("ignore", RuntimeWarning)
        return z["r"], {int(n): np.nanmean(z["B"][z["ntu"] == n], axis=0) for n in sorted(set(z["ntu"]))}


b = bundle()
st.title("Turbidity from laser-spot blur (iPhone Bayer RAW)")
st.caption(f"Model: {b['name']} on `{', '.join(b['features'])}` · calibrated on "
           f"{b['n_independent_samples']} independent samples ({', '.join(map(str, b['train_ntu_levels']))} NTU) · "
           f"resolves ≥ ~{b['detection_floor_ntu']} NTU only")

with st.sidebar:
    st.header("Capture settings used for calibration")
    st.json(b["calib_exif"])
    st.markdown("Shoot the same way: Lightroom RAW (not ProRAW), fixed focus 25 %, dark box, same cuvette position. "
                "Different settings are flagged but not corrected.")

files = st.file_uploader("DNG file(s)", type=["dng"], accept_multiple_files=True)
for up in files or []:
    st.divider()
    with tempfile.TemporaryDirectory() as td:                  # rawpy needs a real file path
        path = Path(td) / up.name
        path.write_bytes(up.getvalue())
        try:
            res, o = analyse(path, b)
            B, _ = load_b_plane(path)
        except Exception as e:
            st.error(f"{up.name}: {e}")
            continue

    st.subheader(up.name)
    c1, c2 = st.columns([1, 2])
    with c1:
        if res["status"] == "rejected":
            st.error(res["message"])
        elif res["below_detection"]:
            st.warning(res["message"])
        else:
            st.metric("Estimated turbidity", f"{res['ntu']:.0f} NTU",
                      help="Range = ±1 sd of leave-one-sample-out error, as a multiplicative factor")
            st.write(f"1-sd range: **{res['ntu_low']:.0f} – {res['ntu_high']:.0f} NTU**")
        for w in res["qc_warnings"]:
            st.warning(w)
        if res["features"]:
            f = res["features"]
            st.write(f"decay length (inner flank): **{f['decay_len_inner']:.1f} px**, "
                     f"core/halo: {f['core_halo_ratio']:.2f}, clipped core out to r ≈ {f['r_first_valid']:.0f} px")
    with c2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].imshow(np.sqrt(np.clip(B, 0, 1)), cmap="gray")
        if res["centre"][0] is not None:
            cy, cx = res["centre"]
            h = PARAMS["half"]
            axes[0].add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h, fill=False, ec="cyan"))
            axes[0].plot(cx, cy, "c+", ms=12)
            axes[0].imshow(np.ma.masked_less(B, 0.98), cmap="autumn", alpha=0.8)
        axes[0].set_title("B plane (sqrt), ROI, clipped = orange", fontsize=9)
        axes[0].axis("off")
        cal = calibration_profiles()
        if cal is not None:
            r, mean_prof = cal
            for n, prof in mean_prof.items():
                axes[1].semilogy(r, prof, lw=1, alpha=0.6, label=f"calib {n} NTU")
        if "profile" in o:
            axes[1].semilogy(o["r"], o["profile"], "k", lw=2, label="this image")
        axes[1].axvspan(*PARAMS["inner_band"], color="orange", alpha=0.1, label="fit band")
        axes[1].set_xlabel("radius (plane px)")
        axes[1].set_title("radial profile vs calibration", fontsize=9)
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with st.expander("raw result (JSON)"):
        st.json(res)
