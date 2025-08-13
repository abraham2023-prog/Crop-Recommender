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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and cache the dataset
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
        with open('model.pkl', 'rb') as f:
            rf = pickle.load(f)
        with open('minmaxscaler.pkl', 'rb') as f:
            mx = pickle.load(f)
        with open('standardscaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        
        # Verify model loaded correctly
        test_input = np.array([[90, 42, 43, 20.88, 82.0, 6.5, 202.94]])
        test_mx = mx.transform(test_input)
        test_sc = sc.transform(test_mx)
        test_pred = rf.predict(test_sc)
        
        if not isinstance(test_pred, np.ndarray):
            raise ValueError("Model prediction failed test")
            
        return rf, mx, sc
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

rf, mx, sc = load_models()

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Make prediction using the same pipeline as training"""
    try:
        # Create input array matching training data format
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Apply the exact same preprocessing
        mx_features = mx.transform(features)
        sc_features = sc.transform(mx_features)
        
        # Make prediction
        prediction = rf.predict(sc_features)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_crop_stats(crop_label):
    """Get statistics for a crop from the dataset"""
    crop_data = crop_df[crop_df['label'] == crop_label]
    if len(crop_data) == 0:
        return None
        
    stats = {
        'N': crop_data['N'].mean(),
        'P': crop_data['P'].mean(),
        'K': crop_data['K'].mean(),
        'temperature': crop_data['temperature'].mean(),
        'humidity': crop_data['humidity'].mean(),
        'ph': crop_data['ph'].mean(),
        'rainfall': crop_data['rainfall'].mean()
    }
    return stats

def main():
    st.title("🌱 Crop Recommendation System")
    st.markdown("Using Random Forest Classifier (Accuracy: 99.09%)")
    
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
            if rf is None or mx is None or sc is None:
                st.error("Models not loaded properly")
            else:
                prediction = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
                if prediction is not None:
                    st.session_state['prediction'] = {
                        'crop': reverse_crop_dict.get(prediction, "Unknown"),
                        'label': prediction,
                        'inputs': {
                            'N': N, 'P': P, 'K': K,
                            'Temperature': temperature,
                            'Humidity': humidity,
                            'pH': ph,
                            'Rainfall': rainfall
                        }
                    }
    
    # Results section
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        crop_name = pred['crop']
        
        st.success(f"## Recommended Crop: {crop_name.title()}")
        
        # Show input vs typical values
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Input Values")
            input_df = pd.DataFrame.from_dict(pred['inputs'], orient='index', columns=['Value'])
            st.dataframe(input_df, use_container_width=True)
        
        with col2:
            st.subheader(f"Typical {crop_name.title()} Conditions")
            stats = get_crop_stats(pred['label'])
            if stats:
                stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Average Value'])
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.warning("No statistics available for this crop")
        
        # Feature importance
        st.subheader("Feature Importance")
        features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis')
        st.pyplot(fig)
        
        # Test different inputs
        with st.expander("Test Different Scenarios"):
            st.write("Try these preset values to see different recommendations:")
            
            test_cases = [
                ("Hot & Dry", [30, 30, 30, 35.0, 30.0, 7.0, 50.0]),
                ("Cool & Wet", [100, 50, 50, 15.0, 90.0, 6.0, 400.0]),
                ("Balanced", [90, 40, 40, 25.0, 70.0, 6.5, 200.0])
            ]
            
            for name, vals in test_cases:
                if st.button(name):
                    test_pred = predict_crop(*vals)
                    if test_pred is not None:
                        st.write(f"{name} → {reverse_crop_dict.get(test_pred, 'Unknown')}")
    
    # Data exploration
    with st.expander("Dataset Information"):
        st.write(f"Total samples: {len(crop_df)}")
        st.write("Crop distribution:")
        st.bar_chart(crop_df['label'].value_counts())
        
        st.write("Feature distributions:")
        feature = st.selectbox("Select feature to view", crop_df.columns[:-1])
        fig, ax = plt.subplots()
        sns.histplot(crop_df[feature], kde=True, ax=ax)
        st.pyplot(fig)

if __name__ == '__main__':
    main()

