"""
app.py — Streamlit Web Application
==================================
A clean UI to predict Islamabad property prices using the automatically
retrained Random Forest model and geospatial features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configurations
st.set_page_config(
    page_title="Islamabad Property Price Predictor",
    page_icon="🏡",
    layout="centered"
)

# Title & Subtitle
st.title("🏡 Islamabad House Price Predictor")
st.markdown("Predict real estate values in Islamabad using an automated live-updating Machine Learning pipeline.")
st.markdown("---")

# Helper function to format prices cleanly in local numbering units
def format_pkr(price_val):
    if price_val >= 10_000_000:
        return f"PKR {price_val / 10_000_000:.2f} Crore"
    else:
        return f"PKR {price_val / 100_000:.2f} Lakh"

# Load the trained artifacts safely
@st.cache_resource
def load_ml_artifacts():
    if not os.path.exists("rf_model.pkl") or not os.path.exists("encoders.pkl"):
        return None, None
    model = joblib.load("rf_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

model, encoders = load_ml_artifacts()

if model is None or encoders is None:
    st.error("🚨 **Model artifacts missing!**")
    st.info("The repository needs `rf_model.pkl` and `encoders.pkl` to run. Please run your training script or wait for the GitHub Actions retraining pipeline to finish.")
else:
    # Extract encoding references
    loc_map = encoders.get("loc_map", {})
    glob_med = encoders.get("glob_med", 15000000)
    spatial_db = encoders.get("spatial_db", {})

    # Sort available locations alphabetically for a clean dropdown
    available_locations = sorted(list(loc_map.keys()))
    if not available_locations:
        available_locations = ["G-13", "F-7", "E-7", "I-11", "Bahria Town", "DHA Phase 2"]

    st.subheader("📋 Enter Property Specifications")

    # Inputs layout split into clean columns
    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("📍 Location", available_locations)
        area_marla = st.number_input("📏 Area (Marla)", min_value=1.0, max_value=500.0, value=7.0, step=0.5)
        bedrooms = st.slider("🛏️ Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.slider("🚿 Bathrooms", min_value=1, max_value=10, value=3)

    with col2:
        kitchens = st.slider("🍳 Kitchens", min_value=0, max_value=5, value=1)
        parking_spaces = st.slider("🚗 Parking Spaces", min_value=0, max_value=5, value=1)
        servant_quarters = st.slider("🧹 Servant Quarters", min_value=0, max_value=5, value=0)
        built_in_year = st.number_input("📅 Built-in Year", min_value=1980, max_value=2026, value=2018, step=1)

    st.markdown("---")

    # Predict Button
    if st.button("💰 Estimate Property Value", use_container_width=True):
        with st.spinner("Calculating valuation..."):
            # 1. Gather inputs into a single row dictionary matching training structure
            input_data = {
                "area_marla": area_marla,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "kitchens": kitchens,
                "parking_spaces": parking_spaces,
                "servant_quarters": servant_quarters,
                "built_in_year": built_in_year,
                "location_clean": location
            }
            
            # 2. Reconstruct the feature engineering matrix exactly like train.py
            X_pred = pd.DataFrame(index=[0])
            X_pred["log_area"] = np.log1p(area_marla)
            X_pred["area_marla"] = area_marla
            X_pred["area_sq"] = area_marla ** 2
            
            # Recreate the exact training area bands
            area_band_val = 6.0   # default for >500 Marla
            for i, b in enumerate([0, 4, 7, 12, 22, 50, 500]):
                if area_marla <= b:
                    area_band_val = float(i)
                    break
            X_pred["area_band"] = area_band_val
            
            # Core specs
            for col in ["parking_spaces", "servant_quarters", "built_in_year", "bedrooms", "bathrooms", "kitchens"]:
                X_pred[col] = float(input_data[col])
                
            # Geospatial distance lookups
            loc_spatial = spatial_db.get(location, {"dist_blue_area": 12.0, "dist_metro": 4.5})
            X_pred["dist_blue_area"] = loc_spatial["dist_blue_area"]
            X_pred["dist_metro"] = loc_spatial["dist_metro"]
            
            # Target Encoding values
            te_val = loc_map.get(location, glob_med)
            X_pred["loc_te"] = te_val
            X_pred["log_loc_te"] = np.log1p(te_val)
            X_pred["area_x_loc"] = area_marla * np.log1p(te_val) / 10.0

            # 3. Predict via Random Forest Model (Target was log-scaled)
            predicted_log_price = model.predict(X_pred)[0]
            estimated_price = np.expm1(predicted_log_price)

            # 4. Generate dynamic upper/lower uncertainty bounds
            lower_bound = estimated_price * 0.90
            upper_bound = estimated_price * 1.10

            # 5. Display the clean output card
            st.success("### 📊 Market Valuation Estimate")
            st.metric(label="Estimated Value", value=format_pkr(estimated_price))
            
            st.markdown(f"""
            * **Expected Market Range:** {format_pkr(lower_bound)} to {format_pkr(upper_bound)}
            * **Location Premium Context:** Model is valuing listings in **{location}** relative to a historical city baseline median of *{format_pkr(glob_med)}*.
            """)
            
st.markdown("---")
st.caption("Powered by GitHub Actions Continuous Training Loop & Random Forest Regressor.")
