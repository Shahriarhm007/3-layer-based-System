import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="SPR Performance Evaluation (3-layer)", layout="wide")
st.title("SPR Performance Evaluation (3-layer)")
st.caption("Select materials, calculate resonance wavelength & performance metrics")

# ---------------------------------
# Constants
# ---------------------------------
FIXED_RI_VALUES = np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42])
MATERIALS = ["Au", "Ag", "Cu", "C"]

# ---------------------------------
# Load Bundles
# ---------------------------------
@st.cache_resource
def load_bundles():
    rlam_bundle = joblib.load("best_xgb_wl_with_preprocessing.pkl")
    fwhm_bundle = joblib.load("best_xgb_fwhm_with_preprocessing.pkl")
    return rlam_bundle, fwhm_bundle

rlam_bundle, fwhm_bundle = load_bundles()

# ---------------------------------
# Helper Functions
# ---------------------------------
def thickness_um(material):
    return 0.035 if material in ("Au", "Ag", "Cu") else 0.00034

def build_aligned_df(ri_values, mat1, mat2, mat3, bundle):
    t1 = thickness_um(mat1)
    t2 = thickness_um(mat2)
    t3 = thickness_um(mat3)

    dist_core_to_2nd = 1.05 + t1
    dist_core_to_3rd = dist_core_to_2nd + t2

    feature_dict = {
        "Analyte RI": ri_values,
        "Material of 1st layer (RIU)": [mat1] * len(ri_values),
        "Material of 2nd layer (RIU)": [mat2] * len(ri_values),
        "Material of 3rd layer (RIU)": [mat3] * len(ri_values),
        "thickness of 1st layer (µm)": t1,
        "thickness of 2nd layer (µm)": t2,
        "thickness of 3rd layer (µm)": t3,
        "Distance bwtn core surface and 2nd layer (µm)": dist_core_to_2nd,
        "Distance bwtn core surface and 3rd layer (µm)": dist_core_to_3rd
    }

    return pd.DataFrame({col: feature_dict.get(col, np.nan) for col in bundle['feature_names']})

def preprocess_for_bundle(raw_df, bundle):
    """Handle both string and numeric encoders, log-transform, and align."""
    df = raw_df.copy()

    for col, le in bundle['label_encoders'].items():
        if col not in df.columns:
            st.error(f"Missing required column '{col}'")
            st.stop()

        if le is None or not hasattr(le, "classes_"):
            st.error(f"Encoder for '{col}' is missing or invalid.")
            st.stop()

        # If encoder expects strings
        if all(isinstance(m, str) for m in le.classes_):
            df[col] = df[col].astype(str).str.strip()
            mapping = {m.lower(): m for m in le.classes_}
            df[col] = df[col].str.lower().map(mapping).fillna(df[col])
            bad_vals = [v for v in df[col].unique() if v not in le.classes_]
            if bad_vals:
                st.error(f"Unexpected values in '{col}': {bad_vals}. Expected: {list(le.classes_)}")
                st.stop()
            df[col] = le.transform(df[col])

        # If encoder expects ints (like your FWHM model)
        elif all(isinstance(m, (np.integer, int)) for m in le.classes_):
            name_to_code = dict(zip(MATERIALS, le.classes_))
            df[col] = df[col].map(name_to_code)
            if df[col].isna().any():
                st.error(f"Could not map materials in '{col}' to {list(le.classes_)}")
                st.stop()

        else:
            st.error(f"Unhandled encoder type for '{col}'")
            st.stop()

    if bundle.get('log_transform', False):
        df = np.log(df + 1e-9)

    return df[bundle['feature_names']]

def predict_rlam_um(ri_values, mat1, mat2, mat3):
    raw_df = build_aligned_df(ri_values, mat1, mat2, mat3, rlam_bundle)
    X_proc = preprocess_for_bundle(raw_df, rlam_bundle)
    return np.exp(rlam_bundle['model'].predict(X_proc))

def predict_fwhm_um(ri_values, mat1, mat2, mat3):
    raw_df = build_aligned_df(ri_values, mat1, mat2, mat3, fwhm_bundle)
    X_proc = preprocess_for_bundle(raw_df, fwhm_bundle)
    return fwhm_bundle['model'].predict(X_proc)

def sensitivity_nm_per_RIU(ri, lam_um):
    lam_nm = lam_um * 1000.0
    dlam = lam_nm[1:] - lam_nm[:-1]
    dn = ri[1:] - ri[:-1]
    return np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn!=0)

def evaluate_metrics(ri, lam_um, fwhm_um):
    S = sensitivity_nm_per_RIU(ri, lam_um)
    if len(S) == 0 or np.all(~np.isfinite(S)):
        return None
    idx = int(np.nanargmax(S))
    S_max = float(S[idx])
    lam_nm = float(lam_um[idx] * 1000.0)
    fwhm_nm = float(fwhm_um[idx] * 1000.0)
    Q = lam_nm / fwhm_nm if fwhm_nm > 0 else np.nan
    FOM = S_max / fwhm_nm if fwhm_nm > 0 else np.nan
    return S, S_max, Q, FOM, ri[idx], lam_nm, fwhm_nm

# ---------------------------------
# UI Controls
# ---------------------------------
m1, m2, m3 = st.columns(3)
with m1:
    mat1 = st.selectbox("Material of 1st Layer", MATERIALS)
with m2:
    mat2 = st.selectbox("Material of 2nd Layer", MATERIALS)
with m3:
    mat3 = st.selectbox("Material of 3rd Layer", MATERIALS)

btn1, btn2 = st.columns(2)
calc_btn = btn1.button("Calculate (R-lam)", type="primary")
eval_btn = btn2.button("Evaluate Performance")

if "table" not in st.session_state:
    st.session_state.table = None
if "lam_um" not in st.session_state:
    st.session_state.lam_um = None
if "fwhm_um" not in st.session_state:
    st.session_state.fwhm_um = None

# ---------------------------------
# Actions
# ---------------------------------
if calc_btn:
    lam_um = predict_rlam_um(FIXED_RI_VALUES, mat1, mat2, mat3)
    fwhm_um = predict_fwhm_um(FIXED_RI_VALUES, mat1, mat2, mat3)

    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um
    st.session_state.table = pd.DataFrame({
        "Analyte RI": FIXED_RI_VALUES,
        "Resonance Wavelength (µm)": lam_um
    })

    st.subheader("R-lam Results")
    st.dataframe(st.session_state.table, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=st.session_state.table.to_csv(index=False).encode(),
        file_name=f"R_lambda_{mat1}-{mat2}-{mat3}.csv",
        mime="text/csv"
    )

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

        S_aligned = np.concatenate([[np.nan], S])
        full = pd.DataFrame({
            "Analyte RI": FIXED_RI_VALUES,
            "Resonance Wavelength (µm)": st.session_state.lam_um,
            "FWHM (µm)": st.session_state.fwhm_um,
            "Sensitivity (nm/RIU)": S_aligned
        })

        st.subheader("Full Results")
        st.dataframe(full, use_container_width=True)

        csv_full = full.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Results CSV",
            data=csv_full,
            file_name=f"Full_Results_{mat1}-{mat2}-{mat3}.csv",
            mime="text/csv"
        )


