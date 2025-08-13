# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# 1. Load dataset and create mappings
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

# 2. Train and save models
def train_models():
    try:
        X = crop_df.drop('label', axis=1)
        y = crop_df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data
        mx = MinMaxScaler().fit(X_train)
        X_train_mx = mx.transform(X_train)
        sc = StandardScaler().fit(X_train_mx)
        X_train_sc = sc.transform(X_train_mx)
        
        # Train model
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_sc, y_train)
        
        # Save models
        with open('model.pkl', 'wb') as f:
            pickle.dump(rf, f, protocol=4)
        with open('minmaxscaler.pkl', 'wb') as f:
            pickle.dump(mx, f, protocol=4)
        with open('standardscaler.pkl', 'wb') as f:
            pickle.dump(sc, f, protocol=4)
            
        return rf, mx, sc
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None

# 3. Load or train models
@st.cache_resource
def load_models():
    # Try to load existing models
    try:
        if all(os.path.exists(f) for f in ['model.pkl', 'minmaxscaler.pkl', 'standardscaler.pkl']):
            with open('model.pkl', 'rb') as f:
                rf = pickle.load(f)
            with open('minmaxscaler.pkl', 'rb') as f:
                mx = pickle.load(f)
            with open('standardscaler.pkl', 'rb') as f:
                sc = pickle.load(f)
            return rf, mx, sc
    except:
        pass
    
    # If loading fails, train new models
    return train_models()

rf, mx, sc = load_models()

# 4. Prediction function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        mx_features = mx.transform(features)
        sc_features = sc.transform(mx_features)
        return rf.predict(sc_features)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# 5. Streamlit app
def main():
    st.title("🌱 Crop Recommendation System")
    
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
            prediction = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            if prediction is not None:
                st.session_state['prediction'] = {
                    'crop': reverse_crop_dict.get(prediction, "Unknown"),
                    'inputs': [N, P, K, temperature, humidity, ph, rainfall]
                }
    
    # Results display
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        st.success(f"## Recommended Crop: {pred['crop'].title()}")
        
        # Feature importance
        st.subheader("Feature Importance")
        features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance.set_index('Feature'))

if __name__ == '__main__':
    main()

