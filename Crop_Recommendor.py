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
# Eritrea-Focused Features
# ----------------------------
@st.cache_data
def load_eritrea_data():
    return pd.DataFrame({
        "Crop": ["Sorghum", "Barley", "Teff", "Maize", "Wheat"],
        "Region": ["Lowlands", "Highlands", "Mid-altitude", "River Valleys", "Highlands"],
        "Latitude": [15.179, 15.423, 15.322, 15.256, 15.401],
        "Longitude": [38.925, 38.847, 38.901, 38.932, 38.812],
        "Min_Temp": [24, 12, 18, 20, 10],
        "Max_Temp": [32, 22, 28, 30, 20],
        "Rainfall": [400, 350, 450, 500, 300],
        "Planting_Season": ["Jun-Jul", "Jul-Aug", "Jun-Jul", "May-Jun", "Jul-Aug"],
        "Harvest_Season": ["Nov-Dec", "Dec-Jan", "Nov-Dec", "Oct-Nov", "Dec-Jan"],
        "Color": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
    })

def show_eritrea_map():
    st.subheader("🇪🇷 Optimal Growing Regions in Eritrea")
    eritrea_df = load_eritrea_data()
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        eritrea_df,
        get_position=["Longitude", "Latitude"],
        get_radius=10000,
        get_fill_color="Color",
        get_line_color=[0, 0, 0],
        pickable=True,
        opacity=0.8
    )
    
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        initial_view_state=pdk.ViewState(
            latitude=15.179,
            longitude=38.925,
            zoom=6,
            pitch=45
        ),
        layers=[layer],
        tooltip={
            "html": """
            <b>Crop:</b> {Crop}<br/>
            <b>Region:</b> {Region}<br/>
            <b>Temp Range:</b> {Min_Temp}°C - {Max_Temp}°C<br/>
            <b>Rainfall:</b> {Rainfall}mm/year<br/>
            <b>Planting:</b> {Planting_Season}<br/>
            <b>Harvest:</b> {Harvest_Season}
            """,
            "style": {
                "backgroundColor": "white",
                "color": "black"
            }
        }
    ))
    
    with st.expander("🗺️ Map Legend"):
        st.markdown("""
        - 🔴 Sorghum (Lowlands)
        - 🟢 Barley (Highlands)  
        - 🔵 Teff (Mid-altitude)
        - 🟡 Maize (River valleys)
        - 🟣 Wheat (High elevations)
        """)

def eritrea_seasonal_calendar():
    st.subheader("🌦️ Eritrea Seasonal Planning")
    eritrea_df = load_eritrea_data()
    
    # Convert season strings to datetime ranges
    planting_months = {
        "Jun-Jul": ("2023-06-01", "2023-07-31"),
        "Jul-Aug": ("2023-07-01", "2023-08-31"),
        "May-Jun": ("2023-05-01", "2023-06-30")
    }
    
    harvest_months = {
        "Nov-Dec": ("2023-11-01", "2023-12-31"),
        "Dec-Jan": ("2023-12-01", "2024-01-31"),
        "Oct-Nov": ("2023-10-01", "2023-11-30")
    }
    
    timeline_data = []
    for _, row in eritrea_df.iterrows():
        timeline_data.append({
            "Crop": row["Crop"],
            "Start": planting_months[row["Planting_Season"]][0],
            "End": planting_months[row["Planting_Season"]][1],
            "Stage": "Planting",
            "Color": row["Color"]
        })
        timeline_data.append({
            "Crop": row["Crop"],
            "Start": harvest_months[row["Harvest_Season"]][0],
            "End": harvest_months[row["Harvest_Season"]][1],
            "Stage": "Harvest",
            "Color": row["Color"]
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df["Start"] = pd.to_datetime(timeline_df["Start"])
    timeline_df["End"] = pd.to_datetime(timeline_df["End"])
    
    fig = px.timeline(
        timeline_df,
        x_start="Start",
        x_end="End",
        y="Crop",
        color="Crop",
        color_discrete_map=dict(zip(eritrea_df["Crop"], eritrea_df["Color"])),
        facet_row="Stage",
        title="Eritrea Crop Calendar"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Original App Features
# ----------------------------
fertilizer_dict = {
    # ... (keep your original fertilizer dictionary)
}

def get_crop_info(crop_name):
    # ... (keep your original crop info function)
    pass

def seasonal_chart(df):
    # ... (keep your original seasonal chart function)
    pass

def batch_predict(df):
    # ... (keep your original batch predict function)
    pass

# ----------------------------
# Streamlit App Layout
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌱 Recommendation", 
    "📖 Crop Info", 
    "💧 Fertilizer Guide",
    "📊 Seasonal Chart",
    "📂 Batch Predict",
    "🇪🇷 Eritrea Focus"
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
# --- New Eritrea-Focused Tab ---
with tab6:
    st.header("Eritrea-Specific Agricultural Tools")
    show_eritrea_map()
    eritrea_seasonal_calendar()
    
    # Add climate summary
    with st.expander("🌡️ Eritrea Climate Overview"):
        st.markdown("""
        **Key Climate Zones:**
        - **Coastal Plain:** Hot and humid (25-35°C)
        - **Western Lowlands:** Hot and arid (30-42°C)
        - **Central Highlands:** Temperate (15-25°C)
        - **Eastern Escarpment:** Variable (20-30°C)
        
        **Rainfall Patterns:**
        - Main rainy season (June-Sept)
        - Short rains (March-April)
        - Annual range: 200-900mm
        """)




