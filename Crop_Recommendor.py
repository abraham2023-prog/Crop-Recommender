# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# 1. Load and prepare data
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

# 2. Model and scaler loading with verification
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
        test_input = np.array([[90, 42, 43, 20.88, 82.0, 6.5, 202.94]])  # Should predict apple
        test_mx = mx.transform(test_input)
        test_sc = sc.transform(test_mx)
        test_pred = rf.predict(test_sc)
        
        if reverse_crop_dict.get(test_pred[0], "") != "apple":
            raise ValueError("Model verification failed")
            
        # Additional verification - should NOT predict apple for coffee conditions
        coffee_test = np.array([[110, 30, 50, 22.0, 85.0, 6.0, 180.0]])
        coffee_pred = rf.predict(sc.transform(mx.transform(coffee_test)))
        if reverse_crop_dict.get(coffee_pred[0], "") == "apple":
            raise ValueError("Model always predicting apple")
            
        return rf, mx, sc
    except Exception as e:
        st.error(f"Model loading failed: {str(e)} - retraining...")
        return retrain_models()

def retrain_models():
    """Retrain models from scratch"""
    try:
        X = crop_df.drop('label', axis=1)
        y = crop_df['label']
        
        mx = MinMaxScaler().fit(X)
        X_mx = mx.transform(X)
        sc = StandardScaler().fit(X_mx)
        X_sc = sc.transform(X_mx)
        
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_sc, y)
        
        # Save with protocol=4
        pickle.dump(rf, open('model.pkl', 'wb'), protocol=4)
        pickle.dump(mx, open('minmaxscaler.pkl', 'wb'), protocol=4)
        pickle.dump(sc, open('standardscaler.pkl', 'wb'), protocol=4)
        
        return rf, mx, sc
    except Exception as e:
        st.error(f"Retraining failed: {str(e)}")
        return None, None, None

rf, mx, sc = load_models()

# 3. Prediction function with debug output
def predict_crop(input_values):
    try:
        st.write("Debug - Input Values:", input_values)
        
        features = np.array([input_values])
        st.write("Debug - Raw Features:", features)
        
        mx_features = mx.transform(features)
        st.write("Debug - After MinMax:", mx_features)
        
        sc_features = sc.transform(mx_features)
        st.write("Debug - After Standard:", sc_features)
        
        prediction = rf.predict(sc_features)
        st.write("Debug - Raw Prediction:", prediction)
        
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# 4. Streamlit app
def main():
    st.title("🌱 Crop Recommendation System")
    st.write("Using RandomForest with automatic error recovery")
    
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
    
    # Debug tests
    with st.expander("🧪 Debug Tests"):
        st.write("Verify model behavior with known inputs:")
        
        if st.button("Test Apple Conditions"):
            pred = predict_crop([90, 42, 43, 20.88, 82.0, 6.5, 202.94])
            st.write("Should be apple:", reverse_crop_dict.get(pred, "Unknown"))
            
        if st.button("Test Coffee Conditions"):
            pred = predict_crop([110, 30, 50, 22.0, 85.0, 6.0, 180.0])
            st.write("Should NOT be apple:", reverse_crop_dict.get(pred, "Unknown"))
    
    # Results display
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        st.success(f"## Recommended Crop: {pred['crop'].title()}")
        
        # Show feature importance
        if rf is not None:
            st.subheader("Feature Importance")
            features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance.set_index('Feature'))

if __name__ == '__main__':
    main()

