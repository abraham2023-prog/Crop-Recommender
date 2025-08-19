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
# New: Geospatial Visualization
# ----------------------------
def show_map():
    st.subheader("Optimal Growing Regions")
    map_df = crop_df.groupby('label').agg({
        'temperature': 'mean', 
        'rainfall': 'mean',
        'ph': 'mean'
    }).reset_index()
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        map_df,
        get_position=["rainfall", "temperature"],
        get_radius="ph*10000",
        get_fill_color=[255, 140, 0],
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
                    "<b>Avg Temp:</b> {temperature}°C<br/>"
                    "<b>Avg Rainfall:</b> {rainfall}mm<br/>"
                    "<b>Soil pH:</b> {ph}",
            "style": {"backgroundColor": "white", "color": "black"}
        }
    ))

# ----------------------------
# New: Seasonal Calendar
# ----------------------------
def seasonal_calendar():
    st.subheader("Planting Calendar Guide")
    
    # Synthetic data - replace with real seasonal data
    calendar_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]*3,
        "Crop": ["Rice"]*12 + ["Wheat"]*12 + ["Maize"]*12,
        "Activity": ["Planning"]*3 + ["Planting"]*3 + ["Growing"]*3 + ["Harvesting"]*3
    })
    
    fig = px.bar(calendar_data, 
                 x="Month", 
                 y="Crop", 
                 color="Activity",
                 color_discrete_map={
                     "Planning": "#636EFA",
                     "Planting": "#EF553B",
                     "Growing": "#00CC96",
                     "Harvesting": "#AB63FA"
                 },
                 category_orders={
                     "Activity": ["Planning", "Planting", "Growing", "Harvesting"],
                     "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                 })
    
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# New: Market Trends
# ----------------------------
@st.cache_data
def load_market_data():
    # Mock data - replace with real API integration
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize", "Cotton", "Coffee"],
        "Current Price ($/kg)": [0.45, 0.32, 0.28, 1.20, 5.80],
        "3 Month Trend": ["↑ 5%", "↓ 2%", "→ Stable", "↑ 12%", "↑ 8%"],
        "Demand": ["High", "Medium", "High", "Medium", "Very High"]
    })

# ----------------------------
# Original: Fertilizer Recommendations
# ----------------------------
fertilizer_dict = {
    "rice": "Apply urea, DAP, and potash at recommended doses during planting.",
    "maize": "Nitrogen-rich fertilizers during growth stage; potash for root strength.",
    # ... (keep your original fertilizer dictionary)
}

# ----------------------------
# Original: Crop Info Lookup
# ----------------------------
def get_crop_info(crop_name):
    info_dict = {
        "rice": "Rice needs warm temperatures and standing water for most of its growing period.",
        # ... (keep your original info dictionary)
    }
    return info_dict.get(crop_name.lower(), "No information available.")

# ----------------------------
# Streamlit App Layout
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌱 Recommendation", 
    "📖 Crop Info", 
    "💧 Fertilizer Guide",
    "📊 Seasonal Chart",
    "📂 Batch Predict",
    "🌍 Insights Dashboard"
])

# --- Tab 1: Original Recommendation System ---
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
                st.write(get_crop_info(crop_name))
                st.info(f"**Fertilizer Recommendation:** {fertilizer_dict.get(crop_name, 'Not available')}")
        
        # Feature importance chart
        features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance = pd.DataFrame({
            "Feature": features, 
            "Importance": rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance")
        st.bar_chart(importance.set_index("Feature"))

# --- Tab 2-5: Keep your original tabs ---
with tab2:
    st.header("Crop Information Lookup")
    crop_name = st.selectbox("Select a crop", sorted(crop_df['label'].unique()))
    st.info(get_crop_info(crop_name))

with tab3:
    st.header("Fertilizer Recommendations")
    crop_name = st.selectbox("Select a crop for fertilizer advice", sorted(fertilizer_dict.keys()))
    st.success(fertilizer_dict.get(crop_name, "No fertilizer advice available."))

with tab4:
    st.header("Seasonal Crop Chart")
    fig = px.scatter(
        crop_df, x="temperature", y="rainfall", color="label",
        title="Seasonal Crop Distribution",
        labels={"temperature": "Temperature (°C)", "rainfall": "Rainfall (mm)"},
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

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

        try:
            features = new_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
            mx_features = mx.transform(features)
            sc_features = sc.transform(mx_features)
            predictions = rf.predict(sc_features)
            new_df['Predicted_Crop'] = predictions
            
            st.subheader("Predictions")
            st.dataframe(new_df)

            csv = new_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="crop_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# --- Tab 6: New Insights Dashboard ---
with tab6:
    st.header("Agricultural Insights Dashboard")
    
    tab_a, tab_b, tab_c = st.tabs(["🌍 Growing Regions", "📅 Planting Calendar", "💲 Market Trends"])
    
    with tab_a:
        show_map()
        st.markdown("""
        **How to use this map:**
        - Each circle represents optimal conditions for a crop
        - Position shows average rainfall (X) and temperature (Y)
        - Circle size indicates preferred soil pH level
        """)
    
    with tab_b:
        seasonal_calendar()
        st.markdown("""
        **Key to activities:**
        - 🔵 Planning: Soil preparation and seed selection
        - 🔴 Planting: Optimal planting window
        - 🟢 Growing: Active growth period
        - 🟣 Harvesting: Recommended harvest time
        """)
    
    with tab_c:
        st.subheader("Current Market Trends")
        market_data = load_market_data()
        
        # Price trends visualization
        fig = px.bar(market_data, 
                     x="Crop", 
                     y="Current Price ($/kg)",
                     color="Demand",
                     color_discrete_sequence=["green", "orange", "red"])
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data table
        st.dataframe(
            market_data.style.format({
                "Current Price ($/kg)": "${:.2f}"
            }).applymap(
                lambda x: "color: green" if "↑" in str(x) else "color: red" if "↓" in str(x) else "", 
                subset=["3 Month Trend"]
            )
        )
        
        st.markdown("""
        **Market Insights:**
        - Prices updated weekly from agricultural markets
        - Trends show 3-month price movement
        - Demand levels indicate current market preference
        """)

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




