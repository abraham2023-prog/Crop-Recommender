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
    "coffee": "Use organic compost, NPK with extra potassium.",
    "apple": "Organic manure plus calcium ammonium nitrate.",
    "chickpea": "Apply phosphorus-rich fertilizers like SSP at sowing; minimal nitrogen required.",
    "kidneybeans": "Balanced NPK with extra phosphorus for root development; avoid excess nitrogen.",
    "pigeonpeas": "Phosphorus-based fertilizers during planting; organic manure improves yield.",
    "mothbeans": "Low nitrogen, moderate phosphorus; respond well to farmyard manure.",
    "mungbean": "Phosphorus-rich fertilizers; avoid excess nitrogen to prevent vegetative growth.",
    "blackgram": "Phosphorus and potassium during sowing; nitrogen only in small doses.",
    "lentil": "Balanced NPK with higher phosphorus; apply gypsum for sulfur needs.",
    "pomegranate": "NPK with extra potassium during fruiting; organic compost for soil health.",
    "banana": "High potassium and nitrogen throughout growth; apply in split doses.",
    "mango": "Farmyard manure plus NPK; extra potassium during flowering and fruiting.",
    "grapes": "NPK with high potassium and magnesium; apply boron to improve fruit set.",
    "watermelon": "Balanced NPK with extra potassium; calcium nitrate improves fruit quality.",
    "muskmelon": "NPK with higher potassium; organic manure for soil structure.",
    "orange": "NPK with emphasis on potassium; micronutrients like zinc and magnesium are beneficial.",
    "papaya": "NPK in equal ratio; magnesium sulfate for leaf health.",
    "coconut": "NPK with extra potassium; magnesium sulfate and organic mulch recommended.",
    "cotton": "Balanced NPK; extra nitrogen during early growth and potassium during boll formation.",
    "jute": "Nitrogen for vegetative growth; phosphorus and potassium for fiber quality."
}

# ----------------------------
# Crop Info Lookup
# ----------------------------
def get_crop_info(crop_name):
    info_dict = {
    "rice": "Rice needs warm temperatures and standing water for most of its growing period.",
    "maize": "Maize prefers well-drained soil and moderate rainfall.",
    "coffee": "Coffee grows in tropical climates with high humidity and moderate shade.",
    "apple": "Apple needs cold winters and mild summers.",
    "chickpea": "Chickpea prefers cool, dry climates and well-drained loamy soils.",
    "kidneybeans": "Kidney beans grow best in warm conditions with moderate rainfall.",
    "pigeonpeas": "Pigeon peas thrive in warm climates and tolerate low rainfall.",
    "mothbeans": "Moth beans are drought-tolerant and grow in sandy, well-drained soils.",
    "mungbean": "Mung beans prefer warm weather and well-drained soils.",
    "blackgram": "Black gram grows well in warm, humid climates with loamy soils.",
    "lentil": "Lentils need cool weather and fertile, well-drained soils.",
    "pomegranate": "Pomegranate thrives in hot, dry climates with low humidity.",
    "banana": "Bananas require warm, humid climates and fertile, well-drained soils.",
    "mango": "Mango trees grow well in tropical and subtropical climates with dry periods.",
    "grapes": "Grapes prefer warm, dry climates with well-drained soils.",
    "watermelon": "Watermelon grows in hot climates and sandy loam soils.",
    "muskmelon": "Muskmelon prefers warm temperatures and sandy, well-drained soils.",
    "orange": "Oranges thrive in subtropical climates with well-drained sandy loam.",
    "papaya": "Papaya grows best in tropical climates with consistent warmth and rainfall.",
    "coconut": "Coconut palms need high humidity, sandy soils, and coastal climates.",
    "cotton": "Cotton grows in warm climates with moderate rainfall and loamy soils.",
    "jute": "Jute requires warm, humid climates with alluvial soils and high rainfall."
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
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Number inputs for NPK with step=1 (whole numbers)
        N = st.number_input('Nitrogen (N)', min_value=0, max_value=150, value=90, step=1)
        P = st.number_input('Phosphorous (P)', min_value=0, max_value=150, value=42, step=1)
        K = st.number_input('Potassium (K)', min_value=0, max_value=150, value=43, step=1)
        
        # Temperature input with number input and range guidance
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=20.88,
            step=0.1,
            help="Typical ranges: 0-15°C (Cool), 15-25°C (Moderate), 25-35°C (Warm), 35-50°C (Hot)"
        )
    
    with col2:
        # Humidity input with regular slider (fixed to match training data range)
        humidity = st.slider(
            'Humidity (%)',
            min_value=0.0,
            max_value=100.0,
            value=82.0,
            step=0.1
        )
        
        # pH input with range indicators
        st.markdown("pH Level")
        ph = st.slider(
            "",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
            label_visibility="collapsed"
        )
        st.markdown("""
        <div style="display: flex; justify-content: space-between; margin-top: -20px">
            <span>Acidic</span>
            <span>Neutral</span>
            <span>Alkaline</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Rainfall input with number input
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=202.94,
            step=1.0,
            help="Typical ranges: <200mm (Arid), 200-400mm (Moderate), >400mm (Wet)"
        )

    if st.button("Recommend Crops", type="primary"):
        top_crops = predict_top_crops(N, P, K, temperature, humidity, ph, rainfall)
        st.success("### Top Recommended Crops:")
        for crop_name, prob in top_crops:
            with st.expander(f"{crop_name.title()} ({prob*100:.1f}% confidence)"):
                st.write(get_crop_info(crop_name))
                st.info(f"**Fertilizer Recommendation:** {fertilizer_dict.get(crop_name, 'Not available')}")

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






