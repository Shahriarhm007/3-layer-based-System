import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="SPR Performance Evaluation (3-layer)", layout="wide")
st.title("SPR Performance Evaluation (3-layer)")
st.caption("Select materials and thicknesses. Click Calculate (R-lam) then Evaluate Performance.")

EPS = 1e-9

@st.cache_resource
def load_model_bundle(path):
    """Load a model bundle containing model, label encoders, feature names, etc."""
    return joblib.load(path)

def thickness_um_nm_input(thick_nm: float) -> float:
    """Convert nanometres to micrometres."""
    return thick_nm / 1000.0

def build_aligned_df(ri_values, mats, thicks_nm, dist, model_bundle):
    """
    Build a DataFrame exactly matching model_bundle['feature_names'],
    encoding categorical materials using the stored label encoders.
    """
    data = {
        "Analyte RI": ri_values,
        "Material of 1st layer (RIU)": model_bundle['label_encoders']['Material of 1st layer (RIU)'].transform([mats[0]])[0],
        "Material of 2nd layer (RIU)": model_bundle['label_encoders']['Material of 2nd layer (RIU)'].transform([mats[1]])[0],
        "Material of 3rd layer (RIU)": model_bundle['label_encoders']['Material of 3rd layer (RIU)'].transform([mats[2]])[0],
        "thickness of 1st layer (µm)": thickness_um_nm_input(thicks_nm[0]),
        "thickness of 2nd layer (µm)": thickness_um_nm_input(thicks_nm[1]),
        "thickness of 3rd layer (µm)": thickness_um_nm_input(thicks_nm[2]),
        "Distance bwtn core surface and 3rd layer (µm)": dist
    }

    df = pd.DataFrame(data)

    # Add any missing features as zeros
    for col in model_bundle['feature_names']:
        if col not in df.columns:
            df[col] = 0

    return df[model_bundle['feature_names']]

def predict_rlam_um(model_bundle, X):
    """Predict R-lam in µm, applying log transform if training used it."""
    if model_bundle.get('log_transform', False):
        X_log = np.log(X + EPS)
        y_log = model_bundle['model'].predict(X_log)
        return np.exp(y_log)
    else:
        return model_bundle['model'].predict(X)

def predict_fwhm_um(model_bundle, X):
    """Predict FWHM in µm."""
    return model_bundle['model'].predict(X)

def sensitivity_nm_per_RIU(ri, lam_um):
    """Calculate sensitivity in nm/RIU."""
    lam_nm = lam_um * 1000.0
    dlam = np.diff(lam_nm)
    dn = np.diff(ri)
    return np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn != 0)

def evaluate_metrics(ri, lam_um, fwhm_um):
    """Compute S_max, Q, FOM, and associated values."""
    S = sensitivity_nm_per_RIU(ri, lam_um)
    if len(S) == 0 or np.all(~np.isfinite(S)):
        return None
    idx_left = int(np.nanargmax(S))
    S_max = float(S[idx_left])
    ri_star = float(ri[idx_left])
    lam_nm_left = float(lam_um[idx_left] * 1000.0)
    fwhm_nm_left = float(fwhm_um[idx_left] * 1000.0)
    Q = lam_nm_left / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    FOM = S_max / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    return dict(S_all=S, S_max=S_max, ri_at_Smax=ri_star,
                lambda_nm_at_Smax_left=lam_nm_left,
                fwhm_nm_at_Smax_left=fwhm_nm_left,
                Q=Q, FOM=FOM, idx_left=idx_left)

# -----------------------------
# UI: Material & thickness input
# -----------------------------
col1, col2, col3 = st.columns(3)
mat1 = col1.selectbox("Plasmonic Metal 1st Layer", ["Au", "Ag", "Cu", "C"])
mat2 = col2.selectbox("Plasmonic Metal 2nd Layer", ["Au", "Ag", "Cu", "C"])
mat3 = col3.selectbox("Plasmonic Metal 3rd Layer", ["Au", "Ag", "Cu", "C"])

th1 = col1.number_input("Thickness 1st (nm)", value=35.0)
th2 = col2.number_input("Thickness 2nd (nm)", value=35.0)
th3 = col3.number_input("Thickness 3rd (nm)", value=35.0)

dist_to_3rd = st.number_input("Distance bwtn core surface and 3rd layer (µm)", value=1.10)

btn1, btn2 = st.columns(2)
calc_btn = btn1.button("Calculate (R-lam)", type="primary")
eval_btn = btn2.button("Evaluate Performance")

if "table" not in st.session_state:
    st.session_state.table = None

# -----------------------------
# Actions
# -----------------------------
if calc_btn:
    rlam_bundle = load_model_bundle("best_xgb_wl_with_preprocessing.pkl")
    fwhm_bundle = load_model_bundle("best_xgb_fwhm_with_preprocessing.pkl")

    # Replace this RI list with your physically‑based lookup if needed
    ri_values = np.linspace(1.33, 1.42, 7)

    X_rlam = build_aligned_df(ri_values, [mat1, mat2, mat3],
                              [th1, th2, th3], dist_to_3rd, rlam_bundle)
    X_fwhm = build_aligned_df(ri_values, [mat1, mat2, mat3],
                              [th1, th2, th3], dist_to_3rd, fwhm_bundle)

    lam_um = predict_rlam_um(rlam_bundle, X_rlam)
    fwhm_um = predict_fwhm_um(fwhm_bundle, X_fwhm)

    st.session_state.ri_values = ri_values
    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um

    table = pd.DataFrame({"Analyte RI": ri_values,
                          "Resonance Wavelength (µm)": lam_um})
    st.session_state.table = table

    st.subheader("R-lam results")
    st.dataframe(table, use_container_width=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (RI vs R-lam)",
                       data=csv,
                       file_name=f"R_lambda_{mat1}-{mat2}-{mat3}.csv",
                       mime="text/csv")

if eval_btn and st.session_state.table is not None:
    metrics = evaluate_metrics(st.session_state.ri_values,
                               st.session_state.lam_um,
                               st.session_state.fwhm_um)
    if metrics:
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Model", f"{mat1}-{mat2}-{mat3}")
        cB.metric("Max. Wavelength Sensitivity", f"{metrics['S_max']:.3f} nm/RIU")
        cC.metric("Q-factor", f"{metrics['Q']:.3f}")
        cD.metric("FOM", f"{metrics['FOM']:.6f}")
        st.caption(
            f"S_max at RI={metrics['ri_at_Smax']:.5f} "
            f"(λ_left={metrics['lambda_nm_at_Smax_left']:.3f} nm, "
            f"FWHM_left={metrics['fwhm_nm_at_Smax_left']:.3f} nm)"
        )
    else:
        st.error("Unable to compute sensitivity. Check RI step and predictions.")
