import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="SPR Performance Evaluation (3-layer)", layout="wide")
st.title("SPR Performance Evaluation (3-layer)")
st.caption("Select three materials → Predict R-λ & FWHM → Evaluate performance (S, S_max, Q, FoM).")

# -----------------------------
# Constants
# -----------------------------
EPS = 1e-9
FIXED_RI_VALUES = np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42], dtype=float)

# Thickness by material (µm)
MATERIAL_THICKNESS = {
    "Au": 0.035,
    "Ag": 0.035,
    "Cu": 0.035,
    "C": 0.00034,
}

ALL_MATERIALS = ["Au", "Ag", "Cu", "C"]

# -----------------------------
# Load bundles (cached)
# -----------------------------
@st.cache_resource
def load_bundle(path):
    """
    Loads a training bundle:
      {
        'model': fitted estimator,
        'label_encoders': {'Material of 1st layer (RIU)': LabelEncoder, ...},
        'feature_names': list of column names used during training,
        'log_transform': True/False  (inputs were log-transformed)
      }
    """
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not a valid training bundle (dict with 'model').")
    return bundle

try:
    wl_bundle = load_bundle("best_xgb_wl_with_preprocessing.pkl")
    fwhm_bundle = load_bundle("best_xgb_fwhm_with_preprocessing.pkl")
except Exception as e:
    st.error(f"Failed to load model bundles: {e}")
    st.stop()

wl_model = wl_bundle["model"]
fwhm_model = fwhm_bundle["model"]

# Prefer encoders from each bundle (they were fit on training materials)
wl_encoders = wl_bundle.get("label_encoders", {})
fwhm_encoders = fwhm_bundle.get("label_encoders", {})

wl_feature_names = wl_bundle.get("feature_names", None)
fwhm_feature_names = fwhm_bundle.get("feature_names", None)

wl_log_input = bool(wl_bundle.get("log_transform", False))
fwhm_log_input = bool(fwhm_bundle.get("log_transform", False))

# -----------------------------
# Helper: encode materials using encoders from bundle
# -----------------------------
def encode_material(encoders_dict, column_name, material):
    """
    encoders_dict: dict like {'Material of 1st layer (RIU)': LabelEncoder, ...}
    column_name: exact feature name used during training for this material column
    material: 'Au' | 'Ag' | 'Cu' | 'C'
    """
    le = encoders_dict.get(column_name, None)
    if le is None:
        # If encoder missing (shouldn't happen), fall back to a simple mapping
        # but warn user because predictions may be off.
        st.warning(f"Missing LabelEncoder for '{column_name}'. Falling back to simple mapping.")
        mapping = {"Au": 0, "Ag": 1, "Cu": 2, "C": 3}
        return mapping[material]
    try:
        return int(le.transform([material])[0])
    except Exception:
        # Material not seen during training; try to handle gracefully
        classes = list(getattr(le, "classes_", []))
        st.error(f"Material '{material}' not in encoder for '{column_name}'. Seen: {classes}")
        st.stop()

# -----------------------------
# Build one feature row matching a bundle's feature_names
# -----------------------------
def build_feature_row_for_bundle(feature_names, encoders_dict, ri, m1, m2, m3):
    """
    Returns a pandas.DataFrame with a single row, columns ordered as feature_names.
    Fills values by recognizing canonical pattern names from training.
    """
    t1 = MATERIAL_THICKNESS[m1]
    t2 = MATERIAL_THICKNESS[m2]
    t3 = MATERIAL_THICKNESS[m3]

    values = {}
    for col in feature_names:
        low = col.lower()
        if "analyte" in low and "ri" in low:
            values[col] = float(ri)
        elif "material" in low and "1st" in low:
            values[col] = encode_material(encoders_dict, col, m1)
        elif "material" in low and "2nd" in low:
            values[col] = encode_material(encoders_dict, col, m2)
        elif "material" in low and "3rd" in low:
            values[col] = encode_material(encoders_dict, col, m3)
        elif "thickness" in low and "1st" in low:
            values[col] = float(t1)
        elif "thickness" in low and "2nd" in low:
            values[col] = float(t2)
        elif "thickness" in low and "3rd" in low:
            values[col] = float(t3)
        else:
            # Unrecognized feature: default to 0.0
            # (If your training had extra engineered features, you can adjust here.)
            values[col] = 0.0

    return pd.DataFrame([values], columns=feature_names)

# -----------------------------
# Predictions across RI grid
# -----------------------------
def predict_series(bundle, model, encoders, mats):
    """
    Predicts values across FIXED_RI_VALUES for a given bundle/model.
    Returns np.array of predictions in ORIGINAL units (i.e. exp(...) if target was log).
    """
    feature_names = bundle["feature_names"]
    log_input = bool(bundle.get("log_transform", False))

    rows = []
    for ri in FIXED_RI_VALUES:
        row_df = build_feature_row_for_bundle(feature_names, encoders, ri, *mats)
        rows.append(row_df)

    X_df = pd.concat(rows, ignore_index=True)

    # If inputs were log-transformed in training, apply here
    X_used = np.log(X_df + EPS) if log_input else X_df

    # Predict; both your WL and FWHM models were trained on log(target),
    # so we exponentiate predictions to get back original units.
    y_log = model.predict(X_used if isinstance(X_used, np.ndarray) else X_used.values)
    y = np.exp(y_log)

    return y  # original units (µm)

# -----------------------------
# Performance metrics
# -----------------------------
def compute_performance(ri, lam_um, fwhm_um):
    """
    Sensitivity S (nm/RIU) = Δλ / Δn, computed between adjacent RI points (λ in nm).
    S_max is the peak S; Q = λ/FWHM at the left RI of S_max; FoM = S_max / FWHM at same point.
    """
    lam_nm = lam_um * 1000.0
    fwhm_nm = fwhm_um * 1000.0

    dlam = lam_nm[1:] - lam_nm[:-1]
    dn = ri[1:] - ri[:-1]
    S = np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn != 0)

    if S.size == 0 or np.all(~np.isfinite(S)):
        return None

    idx_left = int(np.nanargmax(S))
    S_max = float(S[idx_left])

    lam_left_nm = float(lam_nm[idx_left])
    fwhm_left_nm = float(fwhm_nm[idx_left])

    Q = lam_left_nm / fwhm_left_nm if fwhm_left_nm > 0 else np.nan
    FoM = S_max / fwhm_left_nm if fwhm_left_nm > 0 else np.nan

    return {
        "S_all": S,
        "S_max": S_max,
        "ri_at_Smax": float(ri[idx_left]),
        "lambda_nm_at_Smax_left": lam_left_nm,
        "fwhm_nm_at_Smax_left": fwhm_left_nm,
        "Q": Q,
        "FoM": FoM,
        "idx_left": idx_left,
    }

# -----------------------------
# UI: material selection
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    mat1 = st.selectbox("Material (1st layer)", ALL_MATERIALS, index=1)
with c2:
    mat2 = st.selectbox("Material (2nd layer)", ALL_MATERIALS, index=0)
with c3:
    mat3 = st.selectbox("Material (3rd layer)", ALL_MATERIALS, index=2)

st.markdown(
    f"- Thickness(µm): **{mat1}={MATERIAL_THICKNESS[mat1]}**, "
    f"**{mat2}={MATERIAL_THICKNESS[mat2]}**, **{mat3}={MATERIAL_THICKNESS[mat3]}**"
)

mats = (mat1, mat2, mat3)

# Buttons
b1, b2 = st.columns(2)
calc_btn = b1.button("Calculate (R-λ & FWHM)", type="primary")
eval_btn = b2.button("Evaluate Performance")

# Session state
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None

# -----------------------------
# Actions
# -----------------------------
if calc_btn:
    # Predict WL (µm) and FWHM (µm)
    wl_um = predict_series(wl_bundle, wl_model, wl_encoders, mats)
    fwhm_um = predict_series(fwhm_bundle, fwhm_model, fwhm_encoders, mats)

    # Build results table (show nm for readability)
    df = pd.DataFrame({
        "Analyte RI": FIXED_RI_VALUES,
        "Resonance Wavelength (nm)": wl_um * 1000.0,
        "FWHM (nm)": fwhm_um * 1000.0,
    })

    st.session_state.pred_df = df

    st.subheader("Predicted R-λ & FWHM")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (RI vs R-λ & FWHM)",
        data=csv,
        file_name=f"SPR_3layer_{mat1}-{mat2}-{mat3}.csv",
        mime="text/csv",
    )

if eval_btn:
    df = st.session_state.pred_df
    if df is None:
        st.warning("Please click “Calculate (R-λ & FWHM)” first.")
    else:
        # Convert back to µm for metric function (it expects µm and handles nm conversion internally)
        wl_um = df["Resonance Wavelength (nm)"].values / 1000.0
        fwhm_um = df["FWHM (nm)"].values / 1000.0
        ri = df["Analyte RI"].values

        metrics = compute_performance(ri, wl_um, fwhm_um)
        if metrics is None:
            st.error("Unable to compute sensitivity. Check RI spacing and predictions.")
        else:
            mA, mB, mC, mD = st.columns(4)
            mA.metric("Model", f"{mat1}-{mat2}-{mat3}")
            mB.metric("S_max", f"{metrics['S_max']:.3f} nm/RIU")
            mC.metric("Q-factor @ S_max", f"{metrics['Q']:.3f}")
            mD.metric("FoM @ S_max", f"{metrics['FoM']:.6f}")

            st.caption(
                f"S_max at RI = {metrics['ri_at_Smax']:.5f}  •  "
                f"λ_left = {metrics['lambda_nm_at_Smax_left']:.3f} nm  •  "
                f"FWHM_left = {metrics['fwhm_nm_at_Smax_left']:.3f} nm"
            )

            # Optional: show the S curve
            s_df = pd.DataFrame({
                "RI_left": ri[:-1],
                "Sensitivity (nm/RIU)": metrics["S_all"]
            })
            st.line_chart(s_df.set_index("RI_left"))
