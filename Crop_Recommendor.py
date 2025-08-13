# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    crop = pd.read_csv('Crop_recommendation.csv')
    return crop

crop_df = load_data()

# ----------------------------
# Train Model
# ----------------------------
@st.cache_resource
def train_model(data):
    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mx = MinMaxScaler().fit(X_train)
    X_train_mx = mx.transform(X_train)
    sc = StandardScaler().fit(X_train_mx)
    X_train_sc = sc.transform(X_train_mx)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_sc, y_train)

    return rf, mx, sc

rf, mx, sc = train_model(crop_df)

# ----------------------------
# Predict Function
# ----------------------------
def predict_top_crops(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    probs = rf.predict_proba(sc_features)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(rf.classes_[i], probs[i]) for i in top3_idx]

    filtered = []
    for crop, prob in top3:
        if crop == "coffee" and not (18 <= temperature <= 28 and humidity >= 60):
            continue
        if crop == "apple" and not (10 <= temperature <= 20):
            continue
        filtered.append((crop, prob))

    return filtered if filtered else top3

# ----------------------------
# Fertilizer Recommendations
# ----------------------------
fertilizer_dict = {
    "rice": "Apply urea, DAP, and potash at recommended doses during planting.",
    "maize": "Nitrogen-rich fertilizers during growth stage; potash for root strength.",
    "wheat": "Balanced NPK, especially nitrogen during tillering.",
    "coffee": "Use organic compost, NPK with extra potassium.",
    "apple": "Organic manure plus calcium ammonium nitrate."
}

# ----------------------------
# Crop Info Lookup
# ----------------------------
def get_crop_info(crop_name):
    info_dict = {
        "rice": "Rice needs warm temperatures and standing water for most of its growing period.",
        "maize": "Maize prefers well-drained soil and moderate rainfall.",
        "wheat": "Wheat grows well in cool, dry climates.",
        "coffee": "Coffee grows in tropical climates with high humidity and moderate shade.",
        "apple": "Apple needs cold winters and mild summers."
    }
    return info_dict.get(crop_name.lower(), "No information available.")

# ----------------------------
# Seasonal Chart
# ----------------------------
def seasonal_chart(df):
    fig = px.scatter(
        df, x="temperature", y="rainfall", color="label",
        title="Seasonal Crop Distribution",
        labels={"temperature": "Temperature (°C)", "rainfall": "Rainfall (mm)"},
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Batch Prediction
# ----------------------------
def batch_predict(df):
    try:
        features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        mx_features = mx.transform(features)
        sc_features = sc.transform(mx_features)
        predictions = rf.predict(sc_features)
        df['Predicted_Crop'] = predictions
        return df
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# ----------------------------
# Streamlit Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌱 Crop Recommendation",
    "📖 Crop Info",
    "💧 Fertilizer Guide",
    "📊 Seasonal Chart",
    "📂 Upload & Predict"
])

# --- Tab 1: Recommendation ---
with tab1:
    st.header("Get Crop Recommendations")
    N = st.slider('Nitrogen (N)', 0, 150, 90)
    P = st.slider('Phosphorous (P)', 0, 150, 42)
    K = st.slider('Potassium (K)', 0, 150, 43)
    temperature = st.slider('Temperature (°C)', 0.0, 50.0, 20.88)
    humidity = st.slider('Humidity (%)', 0.0, 100.0, 82.0)
    ph = st.slider('pH', 0.0, 14.0, 6.5)
    rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 202.94)

    if st.button("Recommend Crops"):
        top_crops = predict_top_crops(N, P, K, temperature, humidity, ph, rainfall)
        st.success("### Top Recommended Crops:")
        for crop_name, prob in top_crops:
            st.write(f"- **{crop_name.title()}** ({prob*100:.1f}% confidence)")

        features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_})
        importance = importance.sort_values('Importance', ascending=False)
        st.subheader("Feature Importance")
        st.bar_chart(importance.set_index("Feature"))

# --- Tab 2: Crop Info ---
with tab2:
    st.header("Crop Information Lookup")
    crop_name = st.selectbox("Select a crop", sorted(crop_df['label'].unique()))
    st.info(get_crop_info(crop_name))

# --- Tab 3: Fertilizer Guide ---
with tab3:
    st.header("Fertilizer Recommendations")
    crop_name = st.selectbox("Select a crop for fertilizer advice", sorted(fertilizer_dict.keys()))
    st.success(fertilizer_dict.get(crop_name, "No fertilizer advice available."))

# --- Tab 4: Seasonal Chart ---
with tab4:
    st.header("Seasonal Crop Chart")
    seasonal_chart(crop_df)

# --- Tab 5: Upload & Predict ---
with tab5:
    st.header("Upload Dataset for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(new_df.head())

        result_df = batch_predict(new_df)
        if result_df is not None:
            st.subheader("Predictions")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )




