# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

    # Add simple filtering logic for certain crops
    filtered = []
    for crop, prob in top3:
        if crop == "coffee" and not (18 <= temperature <= 28 and humidity >= 60):
            continue
        if crop == "apple" and not (10 <= temperature <= 20):
            continue
        filtered.append((crop, prob))

    return filtered if filtered else top3

# ----------------------------
# Eritrea Data
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
            "style": {"backgroundColor": "white", "color": "black"}
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
    info_dict = {"rice": "Rice needs warm temperatures and standing water for most of its growing period.",
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
# Streamlit App Layout
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌱 Recommendation", "📖 Crop Info", "💧 Fertilizer Guide",
    "📊 Seasonal Chart", "📂 Batch Predict", "🇪🇷 Eritrea Focus"
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

# Soil health analyzer
def soil_health():
    st.subheader("Soil Health Analyzer")
    with st.expander("Nitrogen Deficiency"):
        st.image("https://www.gardeningknowhow.com/wp-content/uploads/2019/07/nitrogen-deficiency.jpg")
        st.write("Symptoms: Yellowing of older leaves, stunted growth")
    
    with st.expander("Phosphorus Deficiency"):
        st.image("https://www.gardeningknowhow.com/wp-content/uploads/2019/07/phosphorus-deficiency.jpg")
        st.write("Symptoms: Dark green leaves with purple discoloration")

# Markrt price
@st.cache_data
def load_market_data():
    # Mock data - replace with real API integration
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Corn"],
        "Current Price": [12.5, 8.2, 4.7],
        "Trend": ["↑ 2%", "↓ 1.5%", "→ Stable"]
    })

def market_trends():
    st.subheader("Market Prices")
    df = load_market_data()
    st.dataframe(df.style.highlight_max(axis=0))
    
    # Add refresh button
    if st.button("Refresh Market Data"):
        st.cache_data.clear()
        df = load_market_data()
# AI chat assistant
from openai import OpenAI

def crop_assistant():
    st.subheader("Crop Advisor Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about crops..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Mock response - replace with real API call
        response = f"Based on your query about '{prompt}', I recommend checking soil nitrogen levels first."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Mobile optimizer
st.markdown("""
<style>
@media screen and (max-width: 600px) {
    .stNumberInput, .stSelectbox {
        width: 100% !important;
    }
    .stButton>button {
        width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)

# User account and History
def user_profile():
    if 'user' not in st.session_state:
        st.text_input("Enter your name to save predictions", key='user_name')
        if st.button("Save Profile"):
            st.session_state.user = st.session_state.user_name
            st.success("Profile saved!")
    else:
        st.success(f"Welcome back {st.session_state.user}!")
        # Add prediction history logic here






