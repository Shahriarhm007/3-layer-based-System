import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model bundles
fwhm_bundle = joblib.load('fwhm_model_bundle.pkl')
angle_bundle = joblib.load('angle_model_bundle.pkl')

# Material options
MATERIALS = ['Au', 'Ag', 'Cu', 'C']

# Streamlit UI
st.title("SPR Sensor Prediction")
st.markdown("Enter the material layers and thicknesses to predict FWHM and Angle.")

col1, col2, col3 = st.columns(3)
mat1 = col1.selectbox("Material of 1st layer (RIU)", MATERIALS)
mat2 = col2.selectbox("Material of 2nd layer (RIU)", MATERIALS)
mat3 = col3.selectbox("Material of 3rd layer (RIU)", MATERIALS)

col4, col5, col6 = st.columns(3)
thick1 = col4.number_input("Thickness of 1st layer (nm)", min_value=1.0)
thick2 = col5.number_input("Thickness of 2nd layer (nm)", min_value=1.0)
thick3 = col6.number_input("Thickness of 3rd layer (nm)", min_value=1.0)

# Prepare input DataFrame
input_df = pd.DataFrame([{
    'Material of 1st layer (RIU)': mat1,
    'Material of 2nd layer (RIU)': mat2,
    'Material of 3rd layer (RIU)': mat3,
    'Thickness of 1st layer (nm)': thick1,
    'Thickness of 2nd layer (nm)': thick2,
    'Thickness of 3rd layer (nm)': thick3
}])

def preprocess(df, bundle):
    df = df.copy()
    encoders = bundle['label_encoders']
    
    # Apply label encoding
    for col, le in encoders.items():
        if all(isinstance(c, str) for c in le.classes_):
            df[col] = df[col].astype(str).str.strip()
            df[col] = le.transform(df[col])
        elif all(isinstance(c, (int, np.integer)) for c in le.classes_):
            name_to_code = dict(zip(MATERIALS, le.classes_))
            df[col] = df[col].map(name_to_code)
    
    # Apply log transform if needed
    if bundle.get('log_transform', False):
        df = df.applymap(lambda x: np.log1p(x) if np.issubdtype(type(x), np.number) else x)
    
    # Align feature order
    df = df[bundle['feature_names']]
    return df

# Predict button
if st.button("Predict"):
    # FWHM prediction
    fwhm_input = preprocess(input_df, fwhm_bundle)
    fwhm_pred = fwhm_bundle['model'].predict(fwhm_input)[0]

    # Angle prediction
    angle_input = preprocess(input_df, angle_bundle)
    angle_pred = angle_bundle['model'].predict(angle_input)[0]

    st.success(f"Predicted FWHM: {fwhm_pred:.4f}")
    st.success(f"Predicted Angle: {angle_pred:.4f}")
