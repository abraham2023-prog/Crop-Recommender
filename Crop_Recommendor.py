# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Crop Recommendation System Pro",
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
# Prediction Functions
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
# Fixed Seasonal Calendar
# ----------------------------
def seasonal_calendar():
    st.subheader("Planting Calendar Guide")
    
    # Corrected with matching array lengths
    months = ["Jan"]*3 + ["Feb"]*3 + ["Mar"]*3 + ["Apr"]*3 + ["May"]*3 + ["Jun"]*3
    crops = ["Rice", "Wheat", "Maize"] * 6
    activities = ["Planning", "Planting", "Harvesting"] * 6
    
    calendar_data = pd.DataFrame({
        "Month": months,
        "Crop": crops,
        "Activity": activities
    })
    
    fig = px.bar(calendar_data, 
                 x="Month", 
                 y="Crop", 
                 color="Activity",
                 color_discrete_map={
                     "Planning": "#636EFA",
                     "Planting": "#EF553B",
                     "Harvesting": "#00CC96"
                 },
                 category_orders={
                     "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                     "Activity": ["Planning", "Planting", "Harvesting"]
                 })
    
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Geospatial Visualization
# ----------------------------
def show_map():
    st.subheader("Optimal Growing Regions")
    
    # Ensure we have the required columns
    if not all(col in crop_df.columns for col in ['temperature', 'rainfall', 'ph', 'label']):
        st.error("Required columns missing in dataset")
        return
    
    map_df = crop_df.groupby('label').agg({
        'temperature': 'mean', 
        'rainfall': 'mean',
        'ph': 'mean'
    }).reset_index()
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        map_df,
        get_position=["rainfall", "temperature"],
        get_radius=100000,
        get_fill_color=[255, 140, 0, 160],
        pickable=True
    )
    
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=15,
            longitude=100,
            zoom=3,
            pitch=50,
        ),
        layers=[layer],
        tooltip={
            "html": "<b>Crop:</b> {label}<br/>"
                    "<b>Avg Temp:</b> {temperature:.1f}°C<br/>"
                    "<b>Avg Rainfall:</b> {rainfall:.1f}mm<br/>"
                    "<b>Soil pH:</b> {ph:.1f}",
            "style": {"backgroundColor": "white", "color": "black"}
        }
    ))

# ----------------------------
# Market Trends
# ----------------------------
@st.cache_data
def load_market_data():
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize"],
        "Current Price ($/kg)": [0.45, 0.32, 0.28],
        "Trend": ["↑ 5%", "↓ 2%", "→ Stable"],
        "Demand": ["High", "Medium", "High"]
    })

# ----------------------------
# Original App Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌱 Recommendation", 
    "📖 Crop Info", 
    "💧 Fertilizer Guide",
    "📊 Seasonal Chart",
    "📂 Batch Predict",
    "🌍 Insights Dashboard"
])

# --- Tab 1: Recommendation System ---
with tab1:
    st.header("Get Crop Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        N = st.number_input('Nitrogen (N)', min_value=0, max_value=150, value=90, step=1)
        P = st.number_input('Phosphorous (P)', min_value=0, max_value=150, value=42, step=1)
        K = st.number_input('Potassium (K)', min_value=0, max_value=150, value=43, step=1)
        temperature = st.number_input('Temperature (°C)', 0.0, 50.0, 20.88, step=0.1)
    
    with col2:
        humidity = st.slider('Humidity (%)', 0.0, 100.0, 82.0, step=0.1)
        ph = st.slider('pH', 0.0, 14.0, 6.5, step=0.1)
        rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 202.94, step=1.0)
    
    if st.button("Recommend Crops", type="primary"):
        top_crops = predict_top_crops(N, P, K, temperature, humidity, ph, rainfall)
        st.success("### Top Recommended Crops:")
        
        for crop_name, prob in top_crops:
            with st.expander(f"{crop_name.title()} ({prob*100:.1f}% confidence)"):
                st.write(f"**Ideal Conditions:**\n\n"
                        f"- Temperature: {crop_df[crop_df['label']==crop_name]['temperature'].mean():.1f}°C\n"
                        f"- Rainfall: {crop_df[crop_df['label']==crop_name]['rainfall'].mean():.1f}mm\n"
                        f"- pH: {crop_df[crop_df['label']==crop_name]['ph'].mean():.1f}")
                
                st.info(f"**Fertilizer Recommendation:**\n\n"
                       f"{fertilizer_dict.get(crop_name, 'Data not available')}")

# --- Tab 6: Insights Dashboard ---
with tab6:
    st.header("Agricultural Insights Dashboard")
    
    tab_a, tab_b, tab_c = st.tabs(["🌍 Growing Regions", "📅 Planting Calendar", "💲 Market Trends"])
    
    with tab_a:
        show_map()
        st.markdown("""
        **Map Interpretation:**
        - Each point shows optimal conditions for different crops
        - Position indicates average rainfall (X) and temperature (Y)
        - Point size represents typical soil pH preference
        """)
    
    with tab_b:
        seasonal_calendar()
        st.markdown("""
        **Calendar Guide:**
        - 🔵 Planning: Soil prep and seed selection
        - 🔴 Planting: Optimal planting window
        - 🟢 Harvesting: Recommended harvest period
        """)
    
    with tab_c:
        st.subheader("Current Market Trends")
        market_data = load_market_data()
        
        fig = px.bar(market_data, 
                     x="Crop", 
                     y="Current Price ($/kg)",
                     color="Demand",
                     color_discrete_map={
                         "High": "green",
                         "Medium": "orange",
                         "Low": "red"
                     })
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            market_data.style.format({
                "Current Price ($/kg)": "${:.2f}"
            }).apply(
                lambda x: ["color: green" if "↑" in v else "color: red" if "↓" in v else "" 
                          for v in x],
                subset=["Trend"]
            )
        )

# Mobile responsive CSS
st.markdown("""
<style>
@media screen and (max-width: 600px) {
    .stNumberInput, .stSelectbox, .stSlider {
        width: 100% !important;
    }
    .stButton>button {
        width: 100%;
    }
    [data-testid="stHorizontalBlock"] {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)




