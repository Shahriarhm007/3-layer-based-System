import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# Load models and encoder
# ------------------------------
wl_model = joblib.load("best_xgb_wl_with_preprocessing.pkl")
fwhm_model = joblib.load("best_xgb_fwhm_with_preprocessing.pkl")
label_encoder = joblib.load("label_encoder_materials.pkl")

# ------------------------------
# Fixed RI values
# ------------------------------
FIXED_RI_VALUES = [1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42]

# ------------------------------
# Material thickness mapping (µm)
# ------------------------------
MATERIAL_THICKNESS = {
    "Au": 0.035,
    "Ag": 0.035,
    "Cu": 0.035,
    "C": 0.00034
}

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("3-Layer SPR System Prediction App")

materials = list(MATERIAL_THICKNESS.keys())

mat1 = st.selectbox("Select Material for Layer 1", materials)
mat2 = st.selectbox("Select Material for Layer 2", materials)
mat3 = st.selectbox("Select Material for Layer 3", materials)

# Encode materials
mat1_enc = label_encoder.transform([mat1])[0]
mat2_enc = label_encoder.transform([mat2])[0]
mat3_enc = label_encoder.transform([mat3])[0]

# Auto thickness assignment
t1 = MATERIAL_THICKNESS[mat1]
t2 = MATERIAL_THICKNESS[mat2]
t3 = MATERIAL_THICKNESS[mat3]

st.write(f"Auto Thickness of {mat1}: {t1} µm")
st.write(f"Auto Thickness of {mat2}: {t2} µm")
st.write(f"Auto Thickness of {mat3}: {t3} µm")

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    predictions_wl = []
    predictions_fwhm = []

    for ri in FIXED_RI_VALUES:
        # Match training feature order: [RI, mat1, t1, mat2, t2, mat3, t3]
        features = np.array([[ri, mat1_enc, t1, mat2_enc, t2, mat3_enc, t3]])

        # Apply inverse log-transform (since training used log-targets)
        wl_pred = float(np.exp(wl_model.predict(features))[0])
        fwhm_pred = float(np.exp(fwhm_model.predict(features))[0])

        predictions_wl.append(wl_pred)
        predictions_fwhm.append(fwhm_pred)

    # ------------------------------
    # Performance Calculations
    # ------------------------------
    wl_array = np.array(predictions_wl)
    fwhm_array = np.array(predictions_fwhm)

    # Sensitivity S = Δλ / Δn
    sensitivities = np.diff(wl_array) / np.diff(FIXED_RI_VALUES)
    Smax = np.max(sensitivities)

    # Q-factor = λ_res / FWHM
    q_factors = wl_array / fwhm_array

    # FoM = Sensitivity / FWHM
    fom_values = sensitivities / fwhm_array[1:]  # align with diff

    # ------------------------------
    # Display Results
    # ------------------------------
    df_results = pd.DataFrame({
        "RI": FIXED_RI_VALUES,
        "Predicted WL (nm)": predictions_wl,
        "Predicted FWHM (nm)": predictions_fwhm,
        "Q-factor": q_factors
    })

    st.subheader("Prediction Results")
    st.dataframe(df_results)

    st.subheader("Performance Metrics")
    st.write(f"**S (per step)**: {sensitivities}")
    st.write(f"**Smax**: {Smax:.2f} nm/RIU")
    st.write(f"**FoM (per step)**: {fom_values}")
    st.write(f"**Average Q-factor**: {np.mean(q_factors):.2f}")

    st.line_chart(df_results.set_index("RI")[["Predicted WL (nm)", "Predicted FWHM (nm)"]])
