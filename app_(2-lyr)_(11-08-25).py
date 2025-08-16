import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="SPR Performance Evaluation (3-layer)", layout="wide")
st.title("SPR Performance Evaluation (3-layer)")
st.caption("Select materials. Click Calculate (R-lam) then Evaluate Performance.")

# -----------------------------
# Constants
# -----------------------------
EPS = 1e-9
# LabelEncoder mapping from your training
ENCODE_MAP = {"Ag": 0, "Au": 1, "C": 2, "Cu": 3}

# -----------------------------
# Helpers
# -----------------------------
def thickness_um(material: str) -> float:
    return 0.035 if material in ("Au", "Ag", "Cu") else 0.00034

FEATURE_COLUMNS = [
    "Analyte RI",
    "Material of 1st layer (RIU)",
    "Material of 2nd layer (RIU)",
    "Material of 3rd layer (RIU)",
    "thickness of 1st layer (µm)",
    "thickness of 2nd layer (µm)",
    "thickness of 3rd layer (µm)",
    "Distance bwtn core surface and 2nd layer (µm)",
    "Distance bwtn core surface and 3rd layer (µm)"
]

@st.cache_resource
def load_models():
    wl_bundle = joblib.load("best_xgb_wl_with_preprocessing.pkl")
    fwhm_bundle = joblib.load("best_xgb_fwhm_with_preprocessing.pkl")
    return wl_bundle, fwhm_bundle

def get_fixed_ri_values():
    # Your fixed analyte RI values
    return np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42])

def build_features(ri_values, mat1, mat2, mat3):
    t1, t2, t3 = thickness_um(mat1), thickness_um(mat2), thickness_um(mat3)
    dist2 = 1.05 + t1        # Distance from core surface to 2nd layer
    dist3 = dist2 + t2       # Distance from core surface to 3rd layer
    df = pd.DataFrame({
        "Analyte RI": ri_values,
        "Material of 1st layer (RIU)": ENCODE_MAP[mat1],
        "Material of 2nd layer (RIU)": ENCODE_MAP[mat2],
        "Material of 3rd layer (RIU)": ENCODE_MAP[mat3],
        "thickness of 1st layer (µm)": t1,
        "thickness of 2nd layer (µm)": t2,
        "thickness of 3rd layer (µm)": t3,
        "Distance bwtn core surface and 2nd layer (µm)": dist2,
        "Distance bwtn core surface and 3rd layer (µm)": dist3
    })
    return df[FEATURE_COLUMNS]

def predict_rlam_um(model_bundle, X_raw):
    X_log = np.log(X_raw + EPS)
    y_log = model_bundle['model'].predict(X_log)
    return np.exp(y_log)

def predict_fwhm_um(model_bundle, X_raw):
    return model_bundle['model'].predict(X_raw)

def sensitivity_nm_per_RIU(ri, lam_um):
    lam_nm = lam_um * 1000.0
    dlam = lam_nm[1:] - lam_nm[:-1]
    dn = ri[1:] - ri[:-1]
    return np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn!=0)

def evaluate_metrics(ri, lam_um, fwhm_um):
    S = sensitivity_nm_per_RIU(ri, lam_um)
    if len(S) == 0 or np.all(~np.isfinite(S)):
        return None
    idx_left = int(np.nanargmax(S))
    lam_nm_left = lam_um[idx_left] * 1000.0
    fwhm_nm_left = fwhm_um[idx_left] * 1000.0
    S_max = S[idx_left]
    ri_star = ri[idx_left]
    Q = lam_nm_left / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    FOM = S_max / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    return {"S_max": S_max, "ri_at_Smax": ri_star, "lambda_nm_at_Smax": lam_nm_left,
            "fwhm_nm_at_Smax": fwhm_nm_left, "Q": Q, "FOM": FOM}

# -----------------------------
# UI: Material selection
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    mat1 = st.selectbox("Plasmonic Metal 1st Layer", ["Ag", "Au", "C", "Cu"], index=1)
with col2:
    mat2 = st.selectbox("Plasmonic Metal 2nd Layer", ["Ag", "Au", "C", "Cu"], index=0)
with col3:
    mat3 = st.selectbox("Plasmonic Metal 3rd Layer", ["Ag", "Au", "C", "Cu"], index=2)

btn1, btn2 = st.columns(2)
calc_btn = btn1.button("Calculate (R-lam)", type="primary")
eval_btn = btn2.button("Evaluate Performance")

# -----------------------------
# Session state
# -----------------------------
if "ri_values" not in st.session_state: st.session_state.ri_values = None
if "lam_um" not in st.session_state: st.session_state.lam_um = None
if "fwhm_um" not in st.session_state: st.session_state.fwhm_um = None
if "table" not in st.session_state: st.session_state.table = None

# -----------------------------
# Actions
# -----------------------------
if calc_btn:
    try:
        wl_bundle, fwhm_bundle = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()
    
    ri_values = get_fixed_ri_values()
    X = build_features(ri_values, mat1, mat2, mat3)
    lam_um = predict_rlam_um(wl_bundle, X)
    fwhm_um = predict_fwhm_um(fwhm_bundle, X)
    
    st.session_state.ri_values = ri_values
    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um
    
    table = pd.DataFrame({"Analyte RI": ri_values, "Resonance Wavelength (µm)": lam_um})
    st.session_state.table = table
    
    st.subheader("R-lam results")
    st.dataframe(table, use_container_width=True)
    st.download_button("Download CSV (RI vs R-lam)", data=table.to_csv(index=False).encode("utf-8"),
                       file_name=f"R_lambda_{mat1}-{mat2}-{mat3}.csv", mime="text/csv")

if eval_btn:
    if st.session_state.table is None:
        st.warning("Please click 'Calculate (R-lam)' first.")
    else:
        metrics = evaluate_metrics(st.session_state.ri_values,
                                   st.session_state.lam_um,
                                   st.session_state.fwhm_um)
        if metrics is None:
            st.error("Unable to compute sensitivity. Check RI step and predictions.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", f"{mat1}-{mat2}-{mat3}")
            c2.metric("Max. Wavelength Sensitivity", f"{metrics['S_max']:.3f} nm/RIU")
            c3.metric("Q-factor", f"{metrics['Q']:.3f}")
            c4.metric("FOM", f"{metrics['FOM']:.6f}")
            st.caption(f"S_max at RI={metrics['ri_at_Smax']:.5f} "
                       f"(λ={metrics['lambda_nm_at_Smax']:.3f} nm, "
                       f"FWHM={metrics['fwhm_nm_at_Smax']:.3f} nm)")

