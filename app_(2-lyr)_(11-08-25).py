import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="SPR Performance Evaluation (3-layer)", layout="wide")
st.title("SPR Performance Evaluation (3-layer)")
st.caption("Select materials. Click Calculate (R-lam & FWHM) then Evaluate Performance.")

# -----------------------------
# Constants and helpers
# -----------------------------
EPS = 1e-9

# Material thickness in µm
def thickness_um(material: str) -> float:
    return 0.035 if material in ("Au", "Ag", "Cu") else 0.00034

# Label encoding mapping from training
LABEL_ENCODER_MAPPING = {
    "Material of 1st layer (RIU)": {"Ag": 0, "Au": 1, "C": 2, "Cu": 3},
    "Material of 2nd layer (RIU)": {"Ag": 0, "Au": 1, "C": 2, "Cu": 3},
    "Material of 3rd layer (RIU)": {"Ag": 0, "Au": 1, "C": 2, "Cu": 3},
}

FEATURE_COLUMNS = [
    "Analyte RI",
    "Material of 1st layer (RIU)",
    "Material of 2nd layer (RIU)",
    "Material of 3rd layer (RIU)",
    "thickness of 1st layer (µm)",
    "thickness of 2nd layer (µm)",
    "thickness of 3rd layer (µm)",
    "Distance bwtn core surface and 2nd layer (µm)",
    "Distance bwtn core surface and 3rd layer (µm)",
]

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    wl_bundle = joblib.load("best_xgb_wl_with_preprocessing.pkl")
    fwhm_bundle = joblib.load("best_xgb_fwhm_with_preprocessing.pkl")
    return wl_bundle, fwhm_bundle

# -----------------------------
# Fixed RI values
# -----------------------------
RI_VALUES = np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42])

# -----------------------------
# Build feature dataframe
# -----------------------------
def build_features(ri_values, mat1, mat2, mat3):
    t1 = thickness_um(mat1)
    t2 = thickness_um(mat2)
    t3 = thickness_um(mat3)
    dist2 = 1.05 + t1
    dist3 = dist2 + t2
    df = pd.DataFrame({
        "Analyte RI": ri_values,
        "Material of 1st layer (RIU)": [LABEL_ENCODER_MAPPING["Material of 1st layer (RIU)"][mat1]]*len(ri_values),
        "Material of 2nd layer (RIU)": [LABEL_ENCODER_MAPPING["Material of 2nd layer (RIU)"][mat2]]*len(ri_values),
        "Material of 3rd layer (RIU)": [LABEL_ENCODER_MAPPING["Material of 3rd layer (RIU)"][mat3]]*len(ri_values),
        "thickness of 1st layer (µm)": t1,
        "thickness of 2nd layer (µm)": t2,
        "thickness of 3rd layer (µm)": t3,
        "Distance bwtn core surface and 2nd layer (µm)": dist2,
        "Distance bwtn core surface and 3rd layer (µm)": dist3,
    })
    return df[FEATURE_COLUMNS]

# -----------------------------
# Prediction functions
# -----------------------------
def predict_wavelength(wl_bundle, X):
    X_log = np.log(X + EPS)
    y_log = wl_bundle['model'].predict(X_log)
    return np.exp(y_log)

def predict_fwhm(fwhm_bundle, X):
    y_log = fwhm_bundle['model'].predict(X)
    return np.exp(y_log)  # inverse log

# -----------------------------
# Performance metrics
# -----------------------------
def sensitivity_nm_per_RIU(ri, lam_um):
    lam_nm = lam_um * 1000.0
    dlam = lam_nm[1:] - lam_nm[:-1]
    dn = ri[1:] - ri[:-1]
    S = np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn!=0)
    return S

def evaluate_metrics(ri, lam_um, fwhm_um):
    S = sensitivity_nm_per_RIU(ri, lam_um)
    if len(S) == 0 or np.all(~np.isfinite(S)):
        return None
    idx = int(np.nanargmax(S))
    S_max = float(S[idx])
    ri_star = float(ri[idx])
    lam_nm_left = float(lam_um[idx] * 1000.0)
    fwhm_nm_left = float(fwhm_um[idx] * 1000.0)
    Q = lam_nm_left / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    FOM = S_max / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    return {
        "S_all": S,
        "S_max": S_max,
        "ri_at_Smax": ri_star,
        "lambda_nm_at_Smax_left": lam_nm_left,
        "fwhm_nm_at_Smax_left": fwhm_nm_left,
        "Q": Q,
        "FOM": FOM,
        "idx": idx
    }

# -----------------------------
# UI: Material selection
# -----------------------------
m1, m2, m3 = st.columns(3)
with m1:
    mat1 = st.selectbox("Plasmonic Metal 1st Layer", ["Au", "Ag", "Cu", "C"], index=1)
with m2:
    mat2 = st.selectbox("Plasmonic Metal 2nd Layer", ["Au", "Ag", "Cu", "C"], index=0)
with m3:
    mat3 = st.selectbox("Plasmonic Metal 3rd Layer", ["Au", "Ag", "Cu", "C"], index=2)

calc_btn, eval_btn = st.columns(2)
calc_btn = calc_btn.button("Calculate (R-lam & FWHM)")
eval_btn = eval_btn.button("Evaluate Performance")

# -----------------------------
# Session state
# -----------------------------
if "ri_values" not in st.session_state: st.session_state.ri_values = None
if "lam_um" not in st.session_state: st.session_state.lam_um = None
if "fwhm_um" not in st.session_state: st.session_state.fwhm_um = None

# -----------------------------
# Actions
# -----------------------------
if calc_btn:
    wl_bundle, fwhm_bundle = load_models()
    X = build_features(RI_VALUES, mat1, mat2, mat3)
    lam_um = predict_wavelength(wl_bundle, X)
    fwhm_um = predict_fwhm(fwhm_bundle, X)
    st.session_state.ri_values = RI_VALUES
    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um

    table = pd.DataFrame({
        "Analyte RI": RI_VALUES,
        "Resonance Wavelength (µm)": lam_um,
        "FWHM (µm)": fwhm_um
    })
    st.subheader("Predicted R-lam & FWHM")
    st.dataframe(table, use_container_width=True)
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"SPR_3layer_{mat1}-{mat2}-{mat3}.csv")

if eval_btn:
    if st.session_state.ri_values is None:
        st.warning("Please calculate predictions first.")
    else:
        metrics = evaluate_metrics(st.session_state.ri_values,
                                   st.session_state.lam_um,
                                   st.session_state.fwhm_um)
        if metrics is None:
            st.error("Unable to compute sensitivity. Check predictions and RI steps.")
        else:
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Model", f"{mat1}-{mat2}-{mat3}")
            colB.metric("Max. Wavelength Sensitivity", f"{metrics['S_max']:.3f} nm/RIU")
            colC.metric("Q-factor", f"{metrics['Q']:.3f}")
            colD.metric("FOM", f"{metrics['FOM']:.6f}")
            st.caption(
                f"S_max at RI={metrics['ri_at_Smax']:.5f} "
                f"(λ_left={metrics['lambda_nm_at_Smax_left']:.3f} nm, "
                f"FWHM_left={metrics['fwhm_nm_at_Smax_left']:.3f} nm)"
            )
