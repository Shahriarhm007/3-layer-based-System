import streamlit as st
import numpy as np
import joblib
import pandas as pd

# -------------------------
# Load Models + Encoder
# -------------------------
@st.cache_resource
def load_models_and_encoder():
    # Load trained models
    wl_model = joblib.load("best_xgb_wl_with_preprocessing.pkl")
    fwhm_model = joblib.load("best_xgb_fwhm_with_preprocessing.pkl")
    # Load shared LabelEncoder
    le = joblib.load("label_encoder_materials.pkl")
    return wl_model, fwhm_model, le

wl_model, fwhm_model, le = load_models_and_encoder()

# -------------------------
# Fixed RI values
# -------------------------
FIXED_RI_VALUES = np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42])

# -------------------------
# Thickness mapping (µm)
# -------------------------
THICKNESS_MAP = {
    "Au": 0.035,
    "Ag": 0.035,
    "Cu": 0.035,
    "C": 0.00034
}

# -------------------------
# Streamlit UI
# -------------------------
st.title("📡 Optical Performance Prediction")
st.markdown("Predict **Resonance Wavelength** and **FWHM** using trained XGBoost models.")

# Material options from encoder
materials = le.classes_

# User selects 3 layers (thickness auto-assigned)
col1, col2, col3 = st.columns(3)
with col1:
    mat1 = st.selectbox("Material of 1st Layer", materials)
with col2:
    mat2 = st.selectbox("Material of 2nd Layer", materials)
with col3:
    mat3 = st.selectbox("Material of 3rd Layer", materials)

# Auto thickness lookup
t1 = THICKNESS_MAP.get(mat1, 0.0)
t2 = THICKNESS_MAP.get(mat2, 0.0)
t3 = THICKNESS_MAP.get(mat3, 0.0)

st.info(f"Auto Thickness (µm): {mat1}={t1}, {mat2}={t2}, {mat3}={t3}")

# -------------------------
# Prediction
# -------------------------
if st.button("🔮 Predict Optical Properties"):
    # Encode materials
    mat1_enc = le.transform([mat1])[0]
    mat2_enc = le.transform([mat2])[0]
    mat3_enc = le.transform([mat3])[0]

    predictions_wl = []
    predictions_fwhm = []

    for ri in FIXED_RI_VALUES:
        # Feature vector (must match training order!)
        features = np.array([
            ri,
            mat1_enc, t1,
            mat2_enc, t2,
            mat3_enc, t3
        ]).reshape(1, -1)

        # FIX: convert safely to scalar
        wl_pred = float(np.exp(wl_model.predict(features)))
        fwhm_pred = float(np.exp(fwhm_model.predict(features)))

        predictions_wl.append(wl_pred)
        predictions_fwhm.append(fwhm_pred)

    # -------------------------
    # Results Table
    # -------------------------
    results_df = pd.DataFrame({
        "Refractive Index (RIU)": FIXED_RI_VALUES,
        "Predicted Resonance Wavelength (µm)": predictions_wl,
        "Predicted FWHM": predictions_fwhm
    })

    st.subheader("📊 Prediction Results")
    st.dataframe(results_df.style.format({
        "Predicted Resonance Wavelength (µm)": "{:.4f}",
        "Predicted FWHM": "{:.4f}"
    }))

    # -------------------------
    # Download Option
    # -------------------------
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
