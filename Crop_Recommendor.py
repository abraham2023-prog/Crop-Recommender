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
    # Load dataset
    crop = pd.read_csv('Crop_recommendation.csv')
    
    # Create mapping (same as your original)
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
@st.cache_resource
def load_models():
    try:
        # Verify and load scalers
        if not os.path.exists('minmaxscaler.pkl') or not os.path.exists('standardscaler.pkl'):
            raise FileNotFoundError("Scaler files missing")
            
        with open('minmaxscaler.pkl', 'rb') as f:
            mx = pickle.load(f)
        with open('standardscaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        
        # Verify and load model
        if not os.path.exists('model.pkl'):
            raise FileNotFoundError("Model file missing")
            
        with open('model.pkl', 'rb') as f:
            rf = pickle.load(f)
        
        # Test prediction with known values
        test_input = np.array([[90, 42, 43, 20.88, 82.0, 6.5, 202.94]])  # Should predict apple
        test_mx = mx.transform(test_input)
        test_sc = sc.transform(test_mx)
        test_pred = rf.predict(test_sc)
        
        if reverse_crop_dict.get(test_pred[0], "") != "apple":
            st.warning("Model test failed - retraining...")
            return retrain_models()
            
        return rf, mx, sc
    except Exception as e:
        st.error(f"Loading failed: {str(e)} - retraining models")
        return retrain_models()

def retrain_models():
    """Fallback to retrain models if loading fails"""
    try:
        X = crop_df.drop('label', axis=1)
        y = crop_df['label']
        
        # Recreate your original preprocessing
        mx = MinMaxScaler().fit(X)
        sc = StandardScaler().fit(mx.transform(X))
        X_transformed = sc.transform(mx.transform(X))
        
        # Retrain RandomForest
        rf = RandomForestClassifier()
        rf.fit(X_transformed, y)
        
        # Save new models
        pickle.dump(rf, open('model.pkl', 'wb'))
        pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
        pickle.dump(sc, open('standardscaler.pkl', 'wb'))
        
        return rf, mx, sc
    except Exception as e:
        st.error(f"Retraining failed: {str(e)}")
        return None, None, None

rf, mx, sc = load_models()

# 3. Prediction function
def predict_crop(input_values):
    try:
        features = np.array([input_values])
        mx_features = mx.transform(features)
        sc_features = sc.transform(mx_features)
        return rf.predict(sc_features)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# 4. Find similar crops from dataset
def find_similar_crops(input_values, main_crop_id, n=5):
    try:
        # Calculate Euclidean distance from input to all samples
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = crop_df[features]
        
        # Scale input the same way
        input_scaled = sc.transform(mx.transform([input_values]))
        X_scaled = sc.transform(mx.transform(X))
        
        # Calculate distances
        distances = np.sqrt(((X_scaled - input_scaled) ** 2).sum(axis=1))
        
        # Get most similar crops (excluding the predicted one)
        similar = crop_df[distances.argsort()]
        similar = similar[similar['label'] != main_crop_id]
        
        return similar.head(n)
    except Exception as e:
        st.error(f"Similar crops error: {str(e)}")
        return pd.DataFrame()

# 5. Streamlit app
def main():
    st.title("🌱 Crop Recommendation System")
    st.write("Model: Random Forest (Retrained)" if rf is None else "Model: Random Forest (Loaded)")
    
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
                    'crop_id': prediction,
                    'crop_name': reverse_crop_dict.get(prediction, "Unknown"),
                    'inputs': input_values
                }
    
    # Results display
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        
        st.success(f"## Recommended Crop: {pred['crop_name'].title()}")
        
        # Show similar crops
        st.subheader("Similar Crops in Dataset")
        similar_crops = find_similar_crops(pred['inputs'], pred['crop_id'])
        
        if not similar_crops.empty:
            # Display similar crops with their parameters
            st.dataframe(
                similar_crops.rename(columns={
                    'temperature': 'Temp',
                    'humidity': 'Humid',
                    'rainfall': 'Rain'
                })[['N', 'P', 'K', 'Temp', 'Humid', 'ph', 'Rain']].assign(
                    Crop=similar_crops['label'].map(reverse_crop_dict)
                ),
                hide_index=True
            )
        else:
            st.warning("No similar crops found in dataset")
        
        # Feature importance
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

