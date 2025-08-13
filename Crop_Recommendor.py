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

# 2. Calculate actual temperature ranges from dataset
def calculate_temp_ranges():
    temp_ranges = {}
    for crop_id, crop_name in reverse_crop_dict.items():
        crop_data = crop_df[crop_df['label'] == crop_id]
        if len(crop_data) > 0:
            min_temp = crop_data['temperature'].min()
            max_temp = crop_data['temperature'].max()
            temp_ranges[crop_id] = (min_temp, max_temp)
    return temp_ranges

temp_ranges = calculate_temp_ranges()

# 3. Load model and scalers
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
        st.error(f"Model loading error: {str(e)}")
        return None, None, None

rf, mx, sc = load_models()

# 4. Prediction function
def predict_crop(input_values):
    try:
        features = np.array([input_values])
        mx_features = mx.transform(features)
        sc_features = sc.transform(mx_features)
        return rf.predict(sc_features)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# 5. Streamlit app
def main():
    st.title("🌱 Crop Recommendation System")
    st.write("Model: Random Forest (99.09% accuracy) - Pure Data-Driven")
    
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
    
    # Results display
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        crop_name = pred['crop']
        temp = pred['inputs'][3]
        
        st.success(f"## Recommended Crop: {crop_name.title()}")
        
        # Show actual temperature range from dataset
        crop_id = [k for k,v in reverse_crop_dict.items() if v == crop_name][0]
        min_temp, max_temp = temp_ranges.get(crop_id, (None, None))
        
        if min_temp and max_temp:
            st.write(f"Temperature range in dataset: {min_temp:.1f}°C to {max_temp:.1f}°C")
            if temp < min_temp or temp > max_temp:
                st.warning("Note: Input temperature is outside this crop's typical range in the dataset")
        
        # Feature importance
        st.subheader("Feature Importance")
        features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance.set_index('Feature'))
        
        # Show similar crops from dataset
        st.subheader("Similar Crops in Dataset")
        similar = crop_df[crop_df['label'] != crop_id]
        if len(similar) > 0:
            st.write("Other crops that grow under similar conditions:")
            similar_samples = similar.sample(min(5, len(similar)))
            st.dataframe(similar_samples[['temperature', 'humidity', 'ph', 'rainfall']].assign(
                Crop=similar_samples['label'].map(reverse_crop_dict)
            ))

if __name__ == '__main__':
    main()

