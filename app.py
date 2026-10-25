# Backward compatibility hotfix for scikit-learn unpickling differences
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import sklearn.impute._base
if not hasattr(sklearn.impute._base.SimpleImputer, "_fill_dtype"):
    sklearn.impute._base.SimpleImputer._fill_dtype = object

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =====================================================================
# 1. PAGE CONFIGURATION & LAYOUT
# =====================================================================
st.set_page_config(
    page_title="Solar Flare Predictor",
    page_icon="☀️",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #d97706; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4b5563; margin-bottom: 25px; }
    .metric-card { background-color: #fef3c7; padding: 20px; border-radius: 10px; border-left: 5px solid #d97706; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Solar Flare Peak Frequency Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Space Weather Forecasting Inference Application Phase (10 Features)</div>', unsafe_allow_html=True)

# =====================================================================
# 2. LOAD TRAINED MODEL PIPELINE
# =====================================================================
MODEL_FILE = "solar_flare_regression_model.pkl"

@st.cache_resource
def load_prediction_pipeline():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_prediction_pipeline()

if pipeline is None:
    st.error(f"⚠️ Error: Final trained pipeline model file `{MODEL_FILE}` was not found in the current directory.")
    st.info("Please make sure you have successfully exported your model using `pickle.dump()` in Step 10.3 of your notebook and uploaded it here.")
    st.stop()

# =====================================================================
# 3. INTERACTIVE APPLICATION INTERFACE (10 INPUTS)
# =====================================================================
st.subheader("📥 Enter Sunspot Active Region Metrics")
st.write("Configure all 10 structural features of the sunspot group below:")

col1, col2 = st.columns(2)

with col1:
    modified_Zurich_class = st.selectbox(
        "Modified Zurich Class:",
        options=["A", "B", "C", "D", "E", "F", "H"],
        index=4,
        help="Structural maturity classification of the active sunspot group."
    )
    
    largest_spot_size = st.selectbox(
        "Largest Spot Size:",
        options=["X", "R", "S", "A", "H", "K"],
        index=5,
        help="Geometric scale and size classification of the principal sunspot core."
    )
    
    spot_distribution = st.selectbox(
        "Spot Distribution:",
        options=["X", "O", "I", "C"],
        index=1,
        help="Structural arrangement and density distribution of spots within the group."
    )
    
    activity = st.selectbox(
        "Activity Index:",
        options=[1, 2],
        format_func=lambda x: "1 = Reduced / Dying" if x == 1 else "2 = Unchanged",
        help="Current structural energy changes over time."
    )
    
    evolution = st.selectbox(
        "Evolution State:",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 = Decay", 2: "2 = No growth", 3: "3 = Growth"}[x],
        index=2,
        help="Operational development phase of the sunspot region."
    )

with col2:
    previous_flare_activity = st.selectbox(
        "Previous 24-Hour Flare Activity:",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 = Nothing as big as an M1", 2: "2 = One M1", 3: "3 = More active than one M1"}[x],
        help="History of recent major eruptions."
    )
    
    historically_complex = st.selectbox(
        "Historically Complex:",
        options=[1, 2],
        format_func=lambda x: "1 = Yes (High Volatility Risk)" if x == 1 else "2 = No",
        help="Tracks if this sunspot region has a tangled magnetic history."
    )
    
    became_complex_on_this_pass = st.selectbox(
        "Became Complex on This Pass:",
        options=[1, 2],
        format_func=lambda x: "1 = Yes (Rapid Accumulation)" if x == 1 else "2 = No",
        help="Indicates if the region grew complex during its current solar disk rotation."
    )
    
    area = st.selectbox(
        "Global Surface Area Scale:",
        options=[1, 2],
        format_func=lambda x: "1 = Small" if x == 1 else "2 = Large",
        index=1,
        help="Total visual footprint footprint scale of the region."
    )
    
    area_of_largest_spot = st.selectbox(
        "Area of Largest Single Spot:",
        options=[1, 2],
        format_func=lambda x: "1 = Less than or equal to 5" if x == 1 else "2 = Greater than 5",
        help="Specific size boundary of the largest single magnetic core."
    )

# =====================================================================
# 4. EXECUTE APPLICATION INFERENCE
# =====================================================================
if st.button("🚀 Calculate Flare Production Rate", type="primary"):
    
    user_input = {
        "modified_Zurich_class": modified_Zurich_class,
        "largest_spot_size": largest_spot_size,
        "spot_distribution": spot_distribution,
        "activity": float(activity),
        "evolution": float(evolution),
        "previous_24_hour_flare_activity": float(previous_flare_activity),
        "historically-complex": float(historically_complex),
        "became_complex_on_this_pass": float(became_complex_on_this_pass),
        "area": float(area),
        "area_of_largest_spot": float(area_of_largest_spot)
    }
    
    # 10 features sorted cleanly by training type arrays
    categorical_features = ["modified_Zurich_class", "largest_spot_size", "spot_distribution"]
    numeric_features = [
        "activity", "evolution", "previous_24_hour_flare_activity", 
        "historically-complex", "became_complex_on_this_pass", "area", "area_of_largest_spot"
    ]
    feature_vector = pd.DataFrame([user_input])[categorical_features + numeric_features]
    
    with st.spinner("Processing architectural features through inference framework..."):
        log_prediction = pipeline.predict(feature_vector)
        predicted_rate = np.expm1(log_prediction)[0]
        predicted_count = int(np.round(predicted_rate))
        if predicted_count < 0:
            predicted_count = 0

    st.markdown("---")
    st.subheader("📊 Space Weather Inference Prediction")
    
    st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color: #b45309;">Model Inference Output:</h4>
            <p style="font-size: 28px; font-weight: bold; margin: 10px 0 5px 0; color: #0b0b0b;">
                {predicted_rate:.2f} <span style="font-size: 18px; font-weight: normal; color: #4b5563;">Flares / 24 Hours</span>
            </p>
            <p style="font-size: 18px; margin: 0; color: #0b0b0b;">
                <strong>Predicted Physical Count:</strong> {predicted_count} {"Flare" if predicted_count == 1 else "Flares"}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption("ℹ️ *Model Evaluation Framework:* This pipeline utilizes a full 10-feature vector mapping layout optimized across a 10-fold cross-validation configuration.")
