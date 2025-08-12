# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Crop dictionary (from your original code)
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

# Model accuracy (replace with your actual accuracy)
MODEL_ACCURACY = 0.99  # 99% accuracy from your original testing

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

def main():
    st.title("🌱 Crop Recommendation System")
    st.markdown(f"*Model Accuracy: {MODEL_ACCURACY*100:.2f}%*")
    
    # Sidebar with input and model info
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
            try:
                prediction = recommendation(N, P, K, temperature, humidity, ph, rainfall)
                crop_name = crop_dict.get(prediction, "Unknown Crop")
                st.session_state['prediction'] = {
                    'crop_name': crop_name,
                    'similar_crops': get_similar_crops(prediction),
                    'inputs': {'N': N, 'P': P, 'K': K, 'Temp': temperature, 
                              'Humidity': humidity, 'pH': ph, 'Rainfall': rainfall}
                }
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        
        # Model info section
        with st.expander("Model Information"):
            st.write(f"**Accuracy:** {MODEL_ACCURACY*100:.2f}%")
            st.write("**Algorithm:** Random Forest Classifier")
            st.write("**Features Used:** 7 soil/weather parameters")

    # Main display area
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            display_crop_image(pred['crop_name'])
            st.dataframe(pd.DataFrame.from_dict(pred['inputs'], orient='index', columns=['Value']), 
                        use_container_width=True)
        
        with col2:
            emoji = crop_emojis.get(pred['crop_name'], '🌱')
            st.success(f"## {emoji} Recommended: {pred['crop_name'].title()}")
            
            # Feature importance plot
            features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            importance = rf.feature_importances_
            fig, ax = plt.subplots()
            sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
        
        # Similar crops
        st.subheader("Similar Crops")
        cols = st.columns(3)
        for idx, crop in enumerate(pred['similar_crops']):
            with cols[idx]:
                with st.container(border=True):
                    st.subheader(f"{crop_emojis.get(crop, '🌱')} {crop.title()}")
                    display_crop_image(crop)
                    st.caption(f"Similar to {pred['crop_name'].title()}")
    else:
        st.info("Adjust parameters and click 'Get Recommendation'")

    # Educational content
    with st.expander("Understanding the Parameters"):
        st.write("""
        - **N (Nitrogen):** Promotes leaf growth (0-150)
        - **P (Phosphorous):** Supports root development (0-150)
        - **K (Potassium):** Enhances fruit quality (0-150)
        - **Temperature:** In Celsius (0-50°C)
        - **Humidity:** Relative humidity (0-100%)
        - **pH:** Soil acidity/alkalinity (0-14)
        - **Rainfall:** Annual precipitation (0-500mm)
        """)

if __name__ == '__main__':
    main()

