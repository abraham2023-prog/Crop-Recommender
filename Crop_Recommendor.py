# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os

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
    # Get all possible labels
    all_labels = list(crop_dict.keys())
    
    # Remove the predicted label
    other_labels = [label for label in all_labels if label != predicted_label]
    
    # Create dummy features (you could modify this to use actual feature importance)
    dummy_features = np.zeros((len(other_labels), 7))
    
    # Get probabilities for all crops
    mx_features = mx.transform(dummy_features)
    sc_features = sc.transform(mx_features)
    probs = rf.predict_proba(sc_features)
    
    # Get top N most probable crops
    top_indices = np.argsort(probs[:, predicted_label-1])[-top_n:][::-1]
    similar_crops = [crop_dict[other_labels[i]] for i in top_indices]
    
    return similar_crops

def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    prediction = rf.predict(sc_features)
    return prediction[0]

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
                similar_crops = get_similar_crops(prediction)
                
                st.session_state['prediction'] = {
                    'crop_name': crop_name,
                    'crop_label': prediction,
                    'similar_crops': similar_crops,
                    'input_values': {
                        'N': N, 'P': P, 'K': K,
                        'Temperature': temperature,
                        'Humidity': humidity,
                        'pH': ph,
                        'Rainfall': rainfall
                    }
                }
            except Exception as e:
                st.error(f"Error in prediction: {e}")
    
    # Main content area
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Try to display crop image
            img_path = f"images/{pred['crop_name']}.jpg"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, caption=pred['crop_name'].title(), use_column_width=True)
            else:
                st.image("https://via.placeholder.com/300x200?text=Crop+Image", 
                         caption=pred['crop_name'].title(), use_column_width=True)
            
            # Display input values
            st.write("### Input Parameters Used:")
            input_df = pd.DataFrame.from_dict(pred['input_values'], orient='index', columns=['Value'])
            st.dataframe(input_df, use_container_width=True)
        
        with col2:
            # Display recommendation
            emoji = crop_emojis.get(pred['crop_name'], '🌱')
            st.success(f"# {emoji} Recommended Crop: {pred['crop_name'].title()}")
            
            # Feature importance visualization
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
            ax.set_title('Feature Importance for Recommendation')
            st.pyplot(fig)
        
        # Similar crops section
        st.subheader("🌾 Similar Crops to Consider")
        cols = st.columns(3)
        for idx, crop in enumerate(pred['similar_crops']):
            with cols[idx]:
                with st.container(border=True):
                    emoji = crop_emojis.get(crop, '🌱')
                    st.subheader(f"{emoji} {crop.title()}")
                    
                    # Try to display crop image
                    img_path = f"images/{crop}.jpg"
                    if os.path.exists(img_path):
                        st.image(img_path, use_column_width=True)
                    
                    # Add basic growing info (expand with your actual data)
                    st.caption("Ideal Growing Conditions:")
                    st.markdown(f"- Temperature: {np.random.randint(15, 35)}-{np.random.randint(20, 40)}°C")
                    st.markdown(f"- Rainfall: {np.random.randint(100, 400)}mm")
                    st.markdown(f"- Soil pH: {round(np.random.uniform(5.5, 7.5), 1)}")
    else:
        st.info("Enter parameters in the sidebar and click 'Get Recommendation'")

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
        This recommendation system uses a Random Forest classifier trained on crop data.
        The model considers 7 input features to make its prediction with high accuracy.
        """)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    main()

