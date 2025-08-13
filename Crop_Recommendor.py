# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# 1. Load dataset
@st.cache_data
def load_data():
    crop = pd.read_csv('Crop_recommendation.csv')
    return crop

crop_df = load_data()

# 2. Train model every time CSV changes
@st.cache_resource
def train_model(data):
    X = data.drop('label', axis=1)
    y = data['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    mx = MinMaxScaler().fit(X_train)
    X_train_mx = mx.transform(X_train)
    sc = StandardScaler().fit(X_train_mx)
    X_train_sc = sc.transform(X_train_mx)

    # Train model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_sc, y_train)

    return rf, mx, sc

rf, mx, sc = train_model(crop_df)

# 3. Prediction with top 3 crops
def predict_top_crops(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    probs = rf.predict_proba(sc_features)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(rf.classes_[i], probs[i]) for i in top3_idx]

    # Apply basic realism filter
    filtered = []
    for crop, prob in top3:
        if crop == "coffee" and not (18 <= temperature <= 28 and humidity >= 60):
            continue
        if crop == "apple" and not (10 <= temperature <= 20):
            continue
        filtered.append((crop, prob))

    return filtered if filtered else top3  # fallback if all filtered out

# 4. Streamlit App
def main():
    st.title("🌱 Crop Recommendation System")
    st.write("Get the top 3 recommended crops based on soil and climate conditions.")

    with st.sidebar:
        st.header("Input Parameters")
        N = st.slider('Nitrogen (N)', 0, 150, 90)
        P = st.slider('Phosphorous (P)', 0, 150, 42)
        K = st.slider('Potassium (K)', 0, 150, 43)
        temperature = st.slider('Temperature (°C)', 0.0, 50.0, 20.88)
        humidity = st.slider('Humidity (%)', 0.0, 100.0, 82.0)
        ph = st.slider('pH', 0.0, 14.0, 6.5)
        rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 202.94)

        if st.button('Get Recommendation'):
            top_crops = predict_top_crops(N, P, K, temperature, humidity, ph, rainfall)
            st.session_state['predictions'] = {
                'crops': top_crops,
                'inputs': [N, P, K, temperature, humidity, ph, rainfall]
            }

    if 'predictions' in st.session_state:
        pred = st.session_state['predictions']
        st.success("### Recommended Crops:")
        for crop_name, prob in pred['crops']:
            st.write(f"- **{crop_name.title()}** ({prob*100:.1f}% confidence)")

        # Feature importance chart
        st.subheader("Feature Importance")
        features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance.set_index('Feature'))

if __name__ == '__main__':
    main()


