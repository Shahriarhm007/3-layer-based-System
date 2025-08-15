import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="SPR Performance Evaluation (3-layer)", layout="wide")
st.title("SPR Performance Evaluation (3-layer)")
st.caption("Select materials for all three layers. Click Calculate (R-lam) then Evaluate Performance.")

# -----------------------------
# Constants and helpers
# -----------------------------
EPS = 1e-9
FIXED_RI_VALUES = np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42])

# Materials in same order as label encoder was trained
# Adjust if your LabelEncoder was fitted with a different ordering
MATERIALS = ["Au", "Ag", "Cu", "C"]

@st.cache_resource
def load_models():
    rlam_model = joblib.load("best_xgboost_model_wl.pkl")
    fwhm_model = joblib.load("best_xgb_model_fwhm.pkl")
    return rlam_model, fwhm_model

def label_encode_material(mat: str) -> int:
    """Mimic the same LabelEncoder mapping used during training."""
    return MATERIALS.index(mat)

FEATURE_COLUMNS = [
    "Analyte RI",
    "Material of 1st layer (RIU)",
    "Material of 2nd layer (RIU)",
    "Material of 3rd layer (RIU)",
    "thickness of 1st layer (µm)",
    "thickness of 2nd layer (µm)",
    "thickness of 3rd layer (µm)",
    "Distance1 (µm)",
    "Distance2 (µm)"
]

def thickness_um(material: str) -> float:
    return 0.035 if material in ("Au", "Ag", "Cu") else 0.00034

def build_features(ri_values, mat1, mat2, mat3) -> pd.DataFrame:
    t1 = thickness_um(mat1)
    t2 = thickness_um(mat2)
    t3 = thickness_um(mat3)
    dist1 = 1.05 + t1
    dist2 = dist1 + t2
    df = pd.DataFrame({
        "Analyte RI": ri_values,
        "Material of 1st layer (RIU)": label_encode_material(mat1),
        "Material of 2nd layer (RIU)": label_encode_material(mat2),
        "Material of 3rd layer (RIU)": label_encode_material(mat3),
        "thickness of 1st layer (µm)": t1,
        "thickness of 2nd layer (µm)": t2,
        "thickness of 3rd layer (µm)": t3,
        "Distance1 (µm)": dist1,
        "Distance2 (µm)": dist2
    })
    return df[FEATURE_COLUMNS]

def predict_rlam_um(rlam_model, X_raw: pd.DataFrame) -> np.ndarray:
    X_log = np.log(X_raw + EPS)
    y_log = rlam_model.predict(X_log)
    return np.exp(y_log)  # invert log-target

def predict_fwhm_um(fwhm_model, X_raw: pd.DataFrame) -> np.ndarray:
    return fwhm_model.predict(X_raw)  # raw target

def sensitivity_nm_per_RIU(ri: np.ndarray, lam_um: np.ndarray) -> np.ndarray:
    lam_nm = lam_um * 1000.0
    dlam = lam_nm[1:] - lam_nm[:-1]
    dn = ri[1:] - ri[:-1]
    return np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn!=0)

def evaluate_metrics(ri: np.ndarray, lam_um: np.ndarray, fwhm_um: np.ndarray):
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
    return {
        "S_all": S,
        "S_max": S_max,
        "ri_at_Smax": ri_star,
        "lambda_nm_at_Smax_left": lam_nm_left,
        "fwhm_nm_at_Smax_left": fwhm_nm_left,
        "Q": Q,
        "FOM": FOM
    }

# -----------------------------
# UI: select materials
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

# Persistent state
if "ri_values" not in st.session_state:
    st.session_state.ri_values = None
if "lam_um" not in st.session_state:
    st.session_state.lam_um = None
if "fwhm_um" not in st.session_state:
    st.session_state.fwhm_um = None
if "table" not in st.session_state:
    st.session_state.table = None

# -----------------------------
# Calculate R-lam
# -----------------------------
if calc_btn:
    try:
        rlam_model, fwhm_model = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    ri_values = FIXED_RI_VALUES
    X = build_features(ri_values, mat1, mat2, mat3)

    lam_um = predict_rlam_um(rlam_model, X)
    fwhm_um = predict_fwhm_um(fwhm_model, X)

    st.session_state.ri_values = ri_values
    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um

    table = pd.DataFrame({
        "Analyte RI": ri_values,
        "Resonance Wavelength (µm)": lam_um
    })
    st.session_state.table = table

    st.subheader("R-lam results")
    st.dataframe(table, use_container_width=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (RI vs R-lam)", data=csv,
                       file_name=f"R_lambda_{mat1}-{mat2}-{mat3}.csv", mime="text/csv")

# -----------------------------
# Evaluate Performance
# -----------------------------
if eval_btn:
    if st.session_state.table is None:
        st.warning("Please click 'Calculate (R-lam)' first.")
    else:
        ri_values = st.session_state.ri_values
        lam_um = st.session_state.lam_um
        fwhm_um = st.session_state.fwhm_um

        metrics = evaluate_metrics(ri_values, lam_um, fwhm_um)
        if metrics is None:
            st.error("Unable to compute sensitivity.")
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

