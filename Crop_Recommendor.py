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

# Initialize fresh scalers (we'll fit them with the dataset)
mx = MinMaxScaler()
sc = StandardScaler()

# Fit scalers with the dataset
X = crop_df.drop('label', axis=1)
mx.fit(X)
sc.fit(mx.transform(X))

# Load only the model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

rf = load_model()

def predict_crop(input_values):
    """Make prediction with proper scaling"""
    try:
        # Transform to numpy array
        features = np.array([input_values])
        
        # Apply transformations
        mx_features = mx.transform(features)
        sc_features = sc.transform(mx_features)
        
        # Make prediction
        prediction = rf.predict(sc_features)
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
    
    # Force different predictions test
    with st.expander("Force Different Predictions"):
        st.write("These inputs should definitely NOT predict apple:")
        
        cols = st.columns(3)
        with cols[0]:
            if st.button("Rice Conditions"):
                pred = predict_crop([83, 45, 60, 28.0, 80.0, 6.5, 250.0])
                st.write(f"Prediction: {reverse_crop_dict.get(pred, 'Unknown')}")
        
        with cols[1]:
            if st.button("Coffee Conditions"):
                pred = predict_crop([110, 30, 50, 22.0, 85.0, 6.0, 180.0])
                st.write(f"Prediction: {reverse_crop_dict.get(pred, 'Unknown')}")
        
        with cols[2]:
            if st.button("Cotton Conditions"):
                pred = predict_crop([90, 60, 50, 30.0, 60.0, 7.5, 100.0])
                st.write(f"Prediction: {reverse_crop_dict.get(pred, 'Unknown')}")
    
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

if __name__ == '__main__':
    main()

