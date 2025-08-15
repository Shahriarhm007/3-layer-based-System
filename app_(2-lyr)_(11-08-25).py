import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="SPR Performance Evaluation (3-layer)", layout="wide")
st.title("SPR Performance Evaluation (3-layer)")
st.caption("Select materials, calculate resonance wavelength & performance metrics")

# -----------------------------
# Constants
# -----------------------------
FIXED_RI_VALUES = np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42])
MATERIALS = ["Au", "Ag", "Cu", "C"]

# -----------------------------
# Load bundles (model + preprocessing)
# -----------------------------
@st.cache_resource
def load_bundles():
    rlam_bundle = joblib.load("best_xgb_wl_with_preprocessing.pkl")
    fwhm_bundle = joblib.load("best_xgb_fwhm_with_preprocessing.pkl")
    return rlam_bundle, fwhm_bundle

rlam_bundle, fwhm_bundle = load_bundles()

# -----------------------------
# Prediction functions
# -----------------------------
def predict_rlam_um(ri_values, mat1, mat2, mat3):
    # Build raw feature DataFrame (names must match training's raw columns)
    t1 = thickness_um(mat1)
    t2 = thickness_um(mat2)
    t3 = thickness_um(mat3)
    dist1 = 1.05 + t1
    dist2 = dist1 + t2

    raw_df = pd.DataFrame({
        "Analyte RI": ri_values,
        "Material of 1st layer (RIU)": [mat1] * len(ri_values),
        "Material of 2nd layer (RIU)": [mat2] * len(ri_values),
        "Material of 3rd layer (RIU)": [mat3] * len(ri_values),
        "thickness of 1st layer (µm)": t1,
        "thickness of 2nd layer (µm)": t2,
        "thickness of 3rd layer (µm)": t3,
        "Distance1 (µm)": dist1,
        "Distance2 (µm)": dist2
    })

    # Directly call the saved pipeline/bundle (handles encoding/log transform)
    lam_um = np.exp(rlam_bundle['model'].predict(
        preprocess_for_bundle(raw_df, rlam_bundle)
    ))
    return lam_um

def predict_fwhm_um(ri_values, mat1, mat2, mat3):
    t1 = thickness_um(mat1)
    t2 = thickness_um(mat2)
    t3 = thickness_um(mat3)
    dist1 = 1.05 + t1
    dist2 = dist1 + t2

    raw_df = pd.DataFrame({
        "Analyte RI": ri_values,
        "Material of 1st layer (RIU)": [mat1] * len(ri_values),
        "Material of 2nd layer (RIU)": [mat2] * len(ri_values),
        "Material of 3rd layer (RIU)": [mat3] * len(ri_values),
        "thickness of 1st layer (µm)": t1,
        "thickness of 2nd layer (µm)": t2,
        "thickness of 3rd layer (µm)": t3,
        "Distance1 (µm)": dist1,
        "Distance2 (µm)": dist2
    })

    fwhm_um = fwhm_bundle['model'].predict(
        preprocess_for_bundle(raw_df, fwhm_bundle)
    )
    return fwhm_um

def thickness_um(material):
    return 0.035 if material in ("Au", "Ag", "Cu") else 0.00034

def preprocess_for_bundle(raw_df, bundle):
    """Apply saved label encoders + log transform according to bundle metadata."""
    df = raw_df.copy()
    for col, le in bundle['label_encoders'].items():
        df[col] = le.transform(df[col])
    if bundle.get('log_transform', False):
        df = np.log(df + 1e-9)
    # Ensure correct column order
    df = df[bundle['feature_names']]
    return df

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
    S_max = float(S[idx_left])
    lam_nm_left = float(lam_um[idx_left] * 1000.0)
    fwhm_nm_left = float(fwhm_um[idx_left] * 1000.0)
    Q = lam_nm_left / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    FOM = S_max / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    return S, S_max, Q, FOM, ri[idx_left], lam_nm_left, fwhm_nm_left

# -----------------------------
# UI
# -----------------------------
m1, m2, m3 = st.columns(3)
with m1:
    mat1 = st.selectbox("Material of 1st Layer", MATERIALS, index=0)
with m2:
    mat2 = st.selectbox("Material of 2nd Layer", MATERIALS, index=1)
with m3:
    mat3 = st.selectbox("Material of 3rd Layer", MATERIALS, index=2)

btn1, btn2 = st.columns(2)
calc_btn = btn1.button("Calculate (R-lam)", type="primary")
eval_btn = btn2.button("Evaluate Performance")

if "table" not in st.session_state:
    st.session_state.table = None
if "lam_um" not in st.session_state:
    st.session_state.lam_um = None
if "fwhm_um" not in st.session_state:
    st.session_state.fwhm_um = None

# -----------------------------
# Actions
# -----------------------------
if calc_btn:
    ri_vals = FIXED_RI_VALUES
    lam_um = predict_rlam_um(ri_vals, mat1, mat2, mat3)
    fwhm_um = predict_fwhm_um(ri_vals, mat1, mat2, mat3)

    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um

    table = pd.DataFrame({
        "Analyte RI": ri_vals,
        "Resonance Wavelength (µm)": lam_um
    })
    st.session_state.table = table

    st.subheader("R-lam Results")
    st.dataframe(table, use_container_width=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv,
                       file_name=f"R_lambda_{mat1}-{mat2}-{mat3}.csv", mime="text/csv")

if eval_btn:
    if st.session_state.table is None:
        st.warning("Please run Calculate (R-lam) first.")
    else:
        S, S_max, Q, FOM, ri_star, lam_nm_star, fwhm_nm_star = evaluate_metrics(
            FIXED_RI_VALUES, st.session_state.lam_um, st.session_state.fwhm_um
        )

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Model", f"{mat1}-{mat2}-{mat3}")
        colB.metric("Max Sensitivity", f"{S_max:.3f} nm/RIU")
        colC.metric("Q-factor", f"{Q:.3f}")
        colD.metric("FOM", f"{FOM:.6f}")

        st.caption(
            f"S_max at RI={ri_star:.5f} "
            f"(λ_left={lam_nm_star:.3f} nm, FWHM_left={fwhm_nm_star:.3f} nm)"
        )

        # Optional full table
        S_aligned = np.concatenate([[np.nan], S])
        full = pd.DataFrame({
            "Analyte RI": FIXED_RI_VALUES,
            "Resonance Wavelength (µm)": st.session_state.lam_um,
            "FWHM (µm)": st.session_state.fwhm_um,
            "Sensitivity (nm/RIU)": S_aligned
        })
        st.subheader("Full Results")
        st.dataframe(full, use_container_width=True)
