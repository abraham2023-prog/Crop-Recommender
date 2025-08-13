# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Load dataset and models
@st.cache_data
def load_data():
    crop_df = pd.read_csv('Crop_recommendation.csv')
    crop_dict = {
        'rice':1, 'maize':2, 'chickpea':3, 'kidneybeans':4,
        'pigeonpeas':5, 'mothbeans':6, 'mungbean':7, 'blackgram':8,
        'lentil':9, 'pomegranate':10, 'banana':11, 'mango':12,
        'grapes':13, 'watermelon':14, 'muskmelon':15, 'apple':16,
        'orange':17, 'papaya':18, 'coconut':19, 'cotton':20,
        'jute':21, 'coffee':22
    }
    crop_df['label'] = crop_df['label'].map(crop_dict)
    return crop_df, {v:k for k,v in crop_dict.items()}

crop_df, reverse_crop_dict = load_data()

@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            rf = pickle.load(f)
        with open('minmaxscaler.pkl', 'rb') as f:
            mx = pickle.load(f)
        with open('standardscaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        return rf, mx, sc
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

rf, mx, sc = load_models()

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Debugged prediction function"""
    try:
        # Debug input values
        st.write("Debug - Input Values:", [N, P, K, temperature, humidity, ph, rainfall])
        
        # Transform using both scalers
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Debug before scaling
        st.write("Debug - Before Scaling:", features)
        
        mx_features = mx.transform(features)
        sc_features = sc.transform(mx_features)
        
        # Debug after scaling
        st.write("Debug - After Scaling:", sc_features)
        
        prediction = rf.predict(sc_features)
        st.write("Debug - Raw Prediction:", prediction)
        
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.title("🌱 Crop Recommendation System")
    
    # Input sliders
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
            prediction = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            if prediction is not None:
                st.session_state['prediction'] = {
                    'crop': reverse_crop_dict.get(prediction, "Unknown"),
                    'label': prediction
                }
    
    # Display results
    if 'prediction' in st.session_state:
        st.success(f"Recommended Crop: {st.session_state['prediction']['crop'].title()}")
        
        # Show feature importance
        if rf is not None:
            st.subheader("Feature Importance")
            features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance.set_index('Feature'))
            
            # Debug - show sample predictions
            st.subheader("Debug Predictions")
            test_values = [
                [90, 42, 43, 20.88, 82.0, 6.5, 202.94],  # Current inputs
                [100, 50, 50, 30.0, 60.0, 7.0, 300.0],    # High temp/rainfall
                [30, 30, 30, 10.0, 90.0, 5.5, 100.0]      # Low values
            ]
            
            for vals in test_values:
                pred = rf.predict(sc.transform(mx.transform([vals])))
                st.write(f"Input {vals} → {reverse_crop_dict.get(pred[0], 'Unknown')}")

if __name__ == '__main__':
    main()

