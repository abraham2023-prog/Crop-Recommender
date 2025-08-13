# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Load dataset and create mappings
@st.cache_data
def load_data():
    crop = pd.read_csv('Crop_recommendation.csv')
    crop_dict = {
        'rice':1, 'maize':2, 'chickpea':3, 'kidneybeans':4,
        'pigeonpeas':5, 'mothbeans':6, 'mungbean':7, 'blackgram':8,
        'lentil':9, 'pomegranate':10, 'banana':11, 'mango':12,
        'grapes':13, 'watermelon':14, 'muskmelon':15, 'apple':16,
        'orange':17, 'papaya':18, 'coconut':19, 'cotton':20,
        'jute':21, 'coffee':22
    }
    crop['label'] = crop['label'].map(crop_dict)
    return crop, {v:k for k,v in crop_dict.items()}

crop_df, reverse_crop_dict = load_data()

# Load models with verification
@st.cache_resource
def load_models():
    try:
        # Load with protocol=4 for compatibility
        with open('model.pkl', 'rb') as f:
            rf = pickle.load(f)
        with open('minmaxscaler.pkl', 'rb') as f:
            mx = pickle.load(f)
        with open('standardscaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        
        # Verify with test prediction
        test_input = np.array([[90, 42, 43, 20.88, 82.0, 6.5, 202.94]])
        transformed = sc.transform(mx.transform(test_input))
        test_pred = rf.predict(transformed)
        
        if test_pred[0] not in reverse_crop_dict:
            raise ValueError("Test prediction invalid")
            
        return rf, mx, sc
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None, None

rf, mx, sc = load_models()

def predict_crop(input_values):
    """Make prediction with debug output"""
    try:
        st.write("Debug - Raw Input:", input_values)
        
        # Transform to numpy array
        features = np.array([input_values])
        st.write("Debug - Array Shape:", features.shape)
        
        # Apply transformations
        mx_features = mx.transform(features)
        st.write("Debug - After MinMax:", mx_features)
        
        sc_features = sc.transform(mx_features)
        st.write("Debug - After Standard:", sc_features)
        
        # Make prediction
        prediction = rf.predict(sc_features)
        st.write("Debug - Prediction:", prediction)
        
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

def main():
    st.title("🌱 Crop Recommendation System")
    st.write("Model: Random Forest (99.09% accuracy)")
    
    # Input section
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
            input_values = [N, P, K, temperature, humidity, ph, rainfall]
            prediction = predict_crop(input_values)
            
            if prediction is not None:
                st.session_state['prediction'] = {
                    'crop': reverse_crop_dict.get(prediction, "Unknown"),
                    'inputs': input_values
                }
    
    # Debug predictions with extreme values
    with st.expander("Debug Tests"):
        st.write("Try these extreme values to verify model behavior:")
        
        if st.button("Test High Temperature (40°C)"):
            pred = predict_crop([90, 42, 43, 40.0, 82.0, 6.5, 202.94])
            st.write("Prediction:", reverse_crop_dict.get(pred, "Unknown"))
            
        if st.button("Test Low Temperature (10°C)"):
            pred = predict_crop([90, 42, 43, 10.0, 82.0, 6.5, 202.94])
            st.write("Prediction:", reverse_crop_dict.get(pred, "Unknown"))
            
        if st.button("Test High Rainfall (400mm)"):
            pred = predict_crop([90, 42, 43, 20.88, 82.0, 6.5, 400.0])
            st.write("Prediction:", reverse_crop_dict.get(pred, "Unknown"))
    
    # Display results
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        st.success(f"## Recommended Crop: {pred['crop'].title()}")
        
        # Show feature importance
        st.subheader("Feature Importance")
        features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance.set_index('Feature'))
        
        # Show input values
        st.subheader("Your Input Values")
        input_df = pd.DataFrame({
            'Feature': features,
            'Value': pred['inputs']
        })
        st.dataframe(input_df.set_index('Feature'))

if __name__ == '__main__':
    main()

