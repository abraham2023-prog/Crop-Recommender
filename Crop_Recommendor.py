# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and scalers
@st.cache_resource
def load_models():
    rf = pickle.load(open('model.pkl', 'rb'))
    mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
    sc = pickle.load(open('standardscaler.pkl', 'rb'))
    return rf, mx, sc

rf, mx, sc = load_models()

# Crop dictionary
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

# Recommendation function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_mx_features = sc.transform(mx_features)
    prediction = rf.predict(sc_mx_features)
    return prediction[0]

# Main app
def main():
    st.title("🌱 Crop Recommendation System")
    st.markdown("""
    This system recommends the best crop to plant based on soil and weather conditions.
    Enter the parameters below to get a recommendation.
    """)
    
    # Sidebar with input parameters
    with st.sidebar:
        st.header("Input Parameters")
        N = st.slider('Nitrogen (N) ratio in soil', 0, 150, 90)
        P = st.slider('Phosphorous (P) ratio in soil', 0, 150, 42)
        K = st.slider('Potassium (K) ratio in soil', 0, 150, 43)
        temperature = st.slider('Temperature (°C)', 0.0, 50.0, 20.88)
        humidity = st.slider('Humidity (%)', 0.0, 100.0, 82.0)
        ph = st.slider('pH value of soil', 0.0, 14.0, 6.5)
        rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 202.94)
        
        if st.button('Get Recommendation'):
            try:
                prediction = recommendation(N, P, K, temperature, humidity, ph, rainfall)
                crop_name = crop_dict.get(prediction, "Unknown Crop")
                st.session_state['prediction'] = crop_name
                st.session_state['input_values'] = {
                    'N': N, 'P': P, 'K': K,
                    'Temperature': temperature,
                    'Humidity': humidity,
                    'pH': ph,
                    'Rainfall': rainfall
                }
            except Exception as e:
                st.error(f"Error in prediction: {e}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Crop Recommendation")
        if 'prediction' in st.session_state:
            st.success(f"Recommended Crop: **{st.session_state['prediction'].upper()}**")
            
            # Display input values
            st.write("### Input Parameters Used:")
            input_df = pd.DataFrame.from_dict(st.session_state['input_values'], orient='index', columns=['Value'])
            st.dataframe(input_df, use_container_width=True)
        else:
            st.info("Enter parameters in the sidebar and click 'Get Recommendation'")
    
    with col2:
        if 'prediction' in st.session_state:
            # Feature importance visualization
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
            ax.set_title('Random Forest Feature Importance')
            st.pyplot(fig)
            
            # Show similar crops (placeholder - you could implement this)
            st.write("### Similar Crops to Consider")
            st.write("""
            - Alternative 1
            - Alternative 2
            - Alternative 3
            """)

    # Add some educational content
    st.markdown("---")
    st.subheader("Understanding Soil Parameters")
    with st.expander("Learn about soil nutrients"):
        st.write("""
        - **Nitrogen (N)**: Essential for leaf growth and green color.
        - **Phosphorous (P)**: Important for root development and flowering.
        - **Potassium (K)**: Helps with overall plant health and disease resistance.
        """)
    
    with st.expander("About the Model"):
        st.write("""
        This recommendation system uses a Random Forest classifier trained on crop data with the following accuracy metrics:
        - Accuracy: ~99%
        - The model considers 7 input features to make its prediction.
        """)

if __name__ == '__main__':
    main()

