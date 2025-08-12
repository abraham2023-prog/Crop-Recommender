# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model accuracy (from your original testing)
MODEL_ACCURACY = 0.99  # 99% accuracy

# Crop dictionary with detailed growing conditions
crop_conditions = {
    'rice': {
        'temp': (20, 30), 
        'rainfall': (150, 300), 
        'ph': (5.0, 7.5), 
        'humidity': (70, 90),
        'N': (50, 100),
        'P': (30, 70),
        'K': (40, 80)
    },
    'maize': {
        'temp': (18, 27), 
        'rainfall': (50, 100), 
        'ph': (5.8, 7.0), 
        'humidity': (60, 80),
        'N': (80, 120),
        'P': (40, 80),
        'K': (50, 90)
    },
    'coffee': {
        'temp': (15, 24), 
        'rainfall': (150, 250), 
        'ph': (6.0, 6.5), 
        'humidity': (70, 90),
        'N': (60, 100),
        'P': (30, 60),
        'K': (40, 80)
    },
    # Add all your crops with their actual conditions
}

# Crop dictionary mapping (from your original code)
crop_dict = {
    1: 'rice',
    2: 'maize',
    3: 'chickpea',
    4: 'kidneybeans',
    5: 'pigeonpeas',
    6: 'mothbeans',
    7: 'mungbean',
    8: 'blackgram',
    9: 'lentil',
    10: 'pomegranate',
    11: 'banana',
    12: 'mango',
    13: 'grapes',
    14: 'watermelon',
    15: 'muskmelon',
    16: 'apple',
    17: 'orange',
    18: 'papaya',
    19: 'coconut',
    20: 'cotton',
    21: 'jute',
    22: 'coffee'
}

# Emoji mapping for crops
crop_emojis = {
    'rice': '🍚',
    'maize': '🌽',
    'chickpea': '🌱',
    'kidneybeans': '🫘',
    'pigeonpeas': '🌱',
    'mothbeans': '🌱',
    'mungbean': '🌱',
    'blackgram': '🌱',
    'lentil': '🌱',
    'pomegranate': '🍈',
    'banana': '🍌',
    'mango': '🥭',
    'grapes': '🍇',
    'watermelon': '🍉',
    'muskmelon': '🍈',
    'apple': '🍎',
    'orange': '🍊',
    'papaya': '🍈',
    'coconut': '🥥',
    'cotton': '🧶',
    'jute': '🌱',
    'coffee': '☕'
}

# Load models and scalers
@st.cache_resource
def load_models():
    rf = pickle.load(open('model.pkl', 'rb'))
    mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
    sc = pickle.load(open('standardscaler.pkl', 'rb'))
    return rf, mx, sc

rf, mx, sc = load_models()

def get_similar_crops(predicted_label, top_n=3):
    """Get similar crops based on model probabilities"""
    all_labels = list(crop_dict.keys())
    other_labels = [label for label in all_labels if label != predicted_label]
    dummy_features = np.zeros((len(other_labels), 7))
    mx_features = mx.transform(dummy_features)
    sc_features = sc.transform(mx_features)
    probs = rf.predict_proba(sc_features)
    top_indices = np.argsort(probs[:, predicted_label-1])[-top_n:][::-1]
    return [crop_dict[other_labels[i]] for i in top_indices]

def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    return rf.predict(sc_features)[0]

def display_crop_image(crop_name):
    """Safely display crop image with fallback to placeholder"""
    try:
        img_path = f"images/{crop_name}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, caption=crop_name.title(), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x200?text=No+Image+Available", 
                    caption=crop_name.title(), use_container_width=True)
    except Exception:
        st.image("https://via.placeholder.com/300x200?text=Image+Error", 
                caption=crop_name.title(), use_container_width=True)

def get_crop_details(crop_name):
    """Return formatted growing conditions for a crop"""
    conditions = crop_conditions.get(crop_name, {})
    return {
        'Temperature': f"{conditions.get('temp', (0,0))[0]}°C - {conditions.get('temp', (0,0))[1]}°C",
        'Rainfall': f"{conditions.get('rainfall', (0,0))[0]}mm - {conditions.get('rainfall', (0,0))[1]}mm",
        'pH': f"{conditions.get('ph', (0,0))[0]} - {conditions.get('ph', (0,0))[1]}",
        'Humidity': f"{conditions.get('humidity', (0,0))[0]}% - {conditions.get('humidity', (0,0))[1]}%",
        'Nitrogen': f"{conditions.get('N', (0,0))[0]} - {conditions.get('N', (0,0))[1]}",
        'Phosphorous': f"{conditions.get('P', (0,0))[0]} - {conditions.get('P', (0,0))[1]}",
        'Potassium': f"{conditions.get('K', (0,0))[0]} - {conditions.get('K', (0,0))[1]}"
    }

def display_similar_crops(similar_crops, main_crop):
    """Display similar crops with detailed growing conditions"""
    st.subheader("🌱 Similar Crops to Consider")
    cols = st.columns(min(3, len(similar_crops)))
    for idx, crop in enumerate(similar_crops):
        with cols[idx]:
            with st.container(border=True):
                emoji = crop_emojis.get(crop, '🌱')
                st.subheader(f"{emoji} {crop.title()}")
                display_crop_image(crop)
                
                # Display detailed conditions
                details = get_crop_details(crop)
                st.markdown("**Ideal Growing Conditions:**")
                st.markdown(f"- 🌡️ Temperature: {details['Temperature']}")
                st.markdown(f"- 🌧️ Rainfall: {details['Rainfall']}")
                st.markdown(f"- 🧪 pH: {details['pH']}")
                st.markdown(f"- 💧 Humidity: {details['Humidity']}")
                st.markdown("**Soil Requirements:**")
                st.markdown(f"- 🟢 Nitrogen (N): {details['Nitrogen']}")
                st.markdown(f"- 🟣 Phosphorous (P): {details['Phosphorous']}")
                st.markdown(f"- 🟠 Potassium (K): {details['Potassium']}")
                
                st.caption(f"Similar to {main_crop.title()}")

def main():
    st.title("🌱 Smart Crop Recommendation System")
    st.markdown(f"*Model Accuracy: {MODEL_ACCURACY*100:.2f}%*")
    
    # Sidebar with input parameters
    with st.sidebar:
        st.header("🌡️ Input Parameters")
        N = st.slider('Nitrogen (N) ratio in soil', 0, 150, 90)
        P = st.slider('Phosphorous (P) ratio in soil', 0, 150, 42)
        K = st.slider('Potassium (K) ratio in soil', 0, 150, 43)
        temperature = st.slider('Temperature (°C)', 0.0, 50.0, 20.88)
        humidity = st.slider('Humidity (%)', 0.0, 100.0, 82.0)
        ph = st.slider('pH value of soil', 0.0, 14.0, 6.5)
        rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 202.94)
        
        if st.button('🌱 Get Recommendation'):
            try:
                prediction = recommendation(N, P, K, temperature, humidity, ph, rainfall)
                crop_name = crop_dict.get(prediction, "Unknown Crop")
                similar_crops = get_similar_crops(prediction)
                
                st.session_state['prediction'] = {
                    'crop_name': crop_name,
                    'crop_label': prediction,
                    'similar_crops': similar_crops,
                    'input_values': {
                        'N': N, 'P': P, 'K': K,
                        'Temperature': temperature,
                        'Humidity': humidity,
                        'pH': ph,
                        'Rainfall': rainfall
                    }
                }
            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
        
        # Model information
        with st.expander("ℹ️ Model Details"):
            st.write(f"**Accuracy:** {MODEL_ACCURACY*100:.2f}%")
            st.write("**Algorithm:** Random Forest Classifier")
            st.write("**Training Data:** 2200 samples")
            st.write("**Features:** 7 soil/weather parameters")

    # Main content area
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display recommended crop image
            display_crop_image(pred['crop_name'])
            
            # Display input parameters used
            st.subheader("📊 Input Parameters Used")
            input_df = pd.DataFrame.from_dict(pred['input_values'], orient='index', columns=['Value'])
            st.dataframe(input_df, use_container_width=True)
        
        with col2:
            # Display recommendation
            emoji = crop_emojis.get(pred['crop_name'], '🌱')
            st.success(f"# {emoji} Recommended Crop: {pred['crop_name'].title()}")
            
            # Display ideal conditions for recommended crop
            st.subheader("🌱 Ideal Growing Conditions")
            details = get_crop_details(pred['crop_name'])
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Temperature:** {details['Temperature']}")
                st.markdown(f"**Rainfall:** {details['Rainfall']}")
            with cols[1]:
                st.markdown(f"**pH Level:** {details['pH']}")
                st.markdown(f"**Humidity:** {details['Humidity']}")
            
            # Feature importance visualization
            st.subheader("📈 Feature Importance")
            features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            importance = rf.feature_importances_
            fig, ax = plt.subplots()
            sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
            ax.set_title('Most Important Factors for This Recommendation')
            st.pyplot(fig)
        
        # Display similar crops with details
        display_similar_crops(pred['similar_crops'], pred['crop_name'])
    else:
        st.info("ℹ️ Adjust the parameters in the sidebar and click 'Get Recommendation'")

    # Educational content
    with st.expander("📚 Understanding the Parameters"):
        st.write("""
        ### Soil Nutrients
        - **Nitrogen (N):** Promotes leaf growth and green color (0-150)
        - **Phosphorous (P):** Supports root development and flowering (0-150)
        - **Potassium (K):** Enhances fruit quality and disease resistance (0-150)
        
        ### Weather Conditions
        - **Temperature:** Ideal growing temperature in Celsius (0-50°C)
        - **Humidity:** Relative humidity percentage (0-100%)
        - **pH:** Soil acidity/alkalinity scale (0-14, 7 is neutral)
        - **Rainfall:** Annual precipitation in millimeters (0-500mm)
        """)

if __name__ == '__main__':
    main()

