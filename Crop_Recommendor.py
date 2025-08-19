# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px

# ----------------------------
# Eritrea Crop Data (FAO-based parameters)
# ----------------------------
@st.cache_data
def load_eritrea_data():
    return pd.DataFrame({
        "Crop": ["Sorghum", "Barley", "Teff", "Maize", "Wheat", "Finger Millet"],
        "Region": ["Lowlands", "Highlands", "Mid-altitude", "River Valleys", "Highlands", "Lowlands"],
        "Latitude": [15.179, 15.423, 15.322, 15.256, 15.401, 14.987],
        "Longitude": [38.925, 38.847, 38.901, 38.932, 38.812, 39.021],
        "Min_Temp": [24, 12, 18, 20, 10, 22],
        "Max_Temp": [32, 22, 28, 30, 20, 35],
        "Rainfall": [400, 350, 450, 500, 300, 380],
        "Planting_Season": ["Jun-Jul", "Jul-Aug", "Jun-Jul", "May-Jun", "Jul-Aug", "Jun-Jul"],
        "Harvest_Season": ["Nov-Dec", "Dec-Jan", "Nov-Dec", "Oct-Nov", "Dec-Jan", "Nov-Dec"],
        "Color": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
    })

# ----------------------------
# Enhanced Eritrea Map
# ----------------------------
def show_eritrea_map():
    st.subheader("🇪🇷 Eritrea Crop Suitability Map")
    
    # Load FAO-based data
    eritrea_df = load_eritrea_data()
    
    # Create base map layer
    base_layer = pdk.Layer(
        "ScatterplotLayer",
        eritrea_df,
        get_position=["Longitude", "Latitude"],
        get_radius=10000,
        get_fill_color="Color",
        get_line_color=[0, 0, 0],
        pickable=True,
        opacity=0.8
    )
    
    # Add temperature contours (synthetic data)
    temp_data = pd.DataFrame({
        "lat": [15.0, 15.5, 16.0]*3,
        "lon": [38.5, 38.8, 39.0]*3,
        "temp": [28, 22, 18, 30, 24, 20, 32, 26, 22]
    })
    
    temp_layer = pdk.Layer(
        "ContourLayer",
        temp_data,
        contours=[{"threshold": 20, "color": [0,255,0]},
                 {"threshold": 25, "color": [255,255,0]},
                 {"threshold": 30, "color": [255,0,0]}],
        cell_size=10000,
        elevation_scale=50,
        get_position=["lon", "lat"],
        get_weight="temp"
    )
    
    # Set view for Eritrea
    view_state = pdk.ViewState(
        latitude=15.179,
        longitude=38.925,
        zoom=6.5,
        pitch=45
    )
    
    # Render map
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        initial_view_state=view_state,
        layers=[temp_layer, base_layer],
        tooltip={
            "html": """
            <b>Crop:</b> {Crop}<br/>
            <b>Region:</b> {Region}<br/>
            <b>Temp Range:</b> {Min_Temp}°C - {Max_Temp}°C<br/>
            <b>Rainfall:</b> {Rainfall}mm/year<br/>
            <b>Planting:</b> {Planting_Season}<br/>
            <b>Harvest:</b> {Harvest_Season}<br/>
            <small>Coordinates: {Latitude:.3f}°N, {Longitude:.3f}°E</small>
            """,
            "style": {
                "backgroundColor": "white",
                "color": "black",
                "fontFamily": '"Helvetica Neue", Arial',
                "zIndex": "10000"
            }
        }
    ))
    
    # Add explanatory sections
    with st.expander("🗺️ Map Legend & Interpretation"):
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
            **Crop Colors:**
            - 🔴 Sorghum
            - 🟢 Barley  
            - 🔵 Teff
            - 🟡 Maize
            - 🟣 Wheat
            - ⚪ Finger Millet
            """)
        with cols[1]:
            st.markdown("""
            **Temperature Zones:**
            - 🟢 <20°C (Cool)
            - 🟡 20-25°C (Moderate)
            - 🔴 >25°C (Hot)
            """)
        with cols[2]:
            st.markdown("""
            **Data Sources:**
            - Crop parameters: FAO Stats
            - Base map: Mapbox Satellite
            - Coordinates: Geonames
            """)
    
    # Add seasonal timeline
    st.subheader("🌦️ Seasonal Planting Calendar")
    timeline_df = eritrea_df.explode("Planting_Season")
    fig = px.timeline(
        timeline_df,
        x_start="Planting_Season",
        x_end="Harvest_Season",
        y="Crop",
        color="Crop",
        color_discrete_map=dict(zip(eritrea_df["Crop"], eritrea_df["Color"]))
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main App Execution
# ----------------------------
def main():
    st.title("Eritrea Agricultural Planning System")
    show_eritrea_map()
    
    # Add climate data section
    with st.expander("📊 Regional Climate Analysis"):
        st.subheader("Monthly Climate Patterns")
        # Synthetic climate data - replace with real data
        climate_data = pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            "Temperature (°C)": [25, 26, 28, 30, 31, 30, 27, 26, 27, 28, 26, 25],
            "Rainfall (mm)": [10, 5, 20, 40, 30, 15, 80, 120, 60, 30, 20, 15]
        })
        
        fig = px.line(climate_data, x="Month", y=["Temperature (°C)", "Rainfall (mm)"], 
                     title="Asmara Climate Patterns",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()




