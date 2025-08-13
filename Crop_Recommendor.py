# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os
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

# Load your original dataset
@st.cache_data
def load_dataset():
    crop = pd.read_csv('Crop_recommendation.csv')
    
    # Create the same mapping you used originally
    crop_dict = {
        'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4,
        'pigeonpeas': 5, 'mothbeans': 6, 'mungbean': 7, 'blackgram': 8,
        'lentil': 9, 'pomegranate': 10, 'banana': 11, 'mango': 12,
        'grapes': 13, 'watermelon': 14, 'muskmelon': 15, 'apple': 16,
        'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
        'jute': 21, 'coffee': 22
    }
    
    crop['label'] = crop['label'].map(crop_dict)
    return crop, {v: k for k, v in crop_dict.items()}

# Load dataset and reverse mapping
crop_df, reverse_crop_dict = load_dataset()

# Load models and scalers
@st.cache_resource
def load_models():
    rf = pickle.load(open('model.pkl', 'rb'))
    mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
    sc = pickle.load(open('standardscaler.pkl', 'rb'))
    return rf, mx, sc

rf, mx, sc = load_models()

def get_similar_crops(predicted_label, top_n=3):
    """Get similar crops based on actual dataset statistics"""
    # Get crops with similar growing conditions from the dataset
    similar = crop_df[crop_df['label'] != predicted_label]
    
    # Calculate similarity based on all features
    target_means = crop_df[crop_df['label'] == predicted_label].mean()
    similar['similarity'] = similar.apply(
        lambda row: 1 / (1 + np.sqrt(
            ((row['N']-target_means['N'])**2 +
            (row['P']-target_means['P'])**2 +
            (row['K']-target_means['K'])**2 +
            (row['temperature']-target_means['temperature'])**2 +
            (row['humidity']-target_means['humidity'])**2 +
            (row['ph']-target_means['ph'])**2 +
            (row['rainfall']-target_means['rainfall'])**2
        )),
        axis=1
    )
    
    # Get top N most similar crops
    top_crops = similar.groupby('label')['similarity'].mean().nlargest(top_n)
    return [reverse_crop_dict[idx] for idx in top_crops.index]

def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    """Make prediction using your original preprocessing pipeline"""
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    prediction = rf.predict(sc_features)
    return prediction[0]

def display_crop_image(crop_name):
    """Safely display crop image with fallback to placeholder"""
    try:
        img_path = f"images/{crop_name}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, caption=crop_name.title(), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x200?text=No+Image+Available", 
                    caption=crop_name.title(), use_container_width=True)
    except Exception:
        st.image("https://via.placeholder.com/300x200?text=Image+Error", 
                caption=crop_name.title(), use_container_width=True)

def get_crop_stats(crop_name):
    """Get actual statistics from your dataset for a crop"""
    if crop_name not in reverse_crop_dict.values():
        return {}
    
    crop_id = [k for k, v in reverse_crop_dict.items() if v == crop_name][0]
    crop_data = crop_df[crop_df['label'] == crop_id]
    
    return {
        'Temperature': f"{crop_data['temperature'].mean():.1f}°C (±{crop_data['temperature'].std():.1f})",
        'Rainfall': f"{crop_data['rainfall'].mean():.1f}mm (±{crop_data['rainfall'].std():.1f})",
        'pH': f"{crop_data['ph'].mean():.1f} (±{crop_data['ph'].std():.1f})",
        'Humidity': f"{crop_data['humidity'].mean():.1f}% (±{crop_data['humidity'].std():.1f})",
        'Nitrogen': f"{crop_data['N'].mean():.1f} (±{crop_data['N'].std():.1f})",
        'Phosphorous': f"{crop_data['P'].mean():.1f} (±{crop_data['P'].std():.1f})",
        'Potassium': f"{crop_data['K'].mean():.1f} (±{crop_data['K'].std():.1f})"
    }

def display_similar_crops(similar_crops, main_crop):
    """Display similar crops with actual dataset statistics"""
    st.subheader("🌱 Similar Crops to Consider")
    cols = st.columns(min(3, len(similar_crops)))
    for idx, crop in enumerate(similar_crops):
        with cols[idx]:
            with st.container(border=True):
                st.subheader(f"🌱 {crop.title()}")
                display_crop_image(crop)
                
                # Display actual statistics from dataset
                stats = get_crop_stats(crop)
                st.markdown("**Growing Conditions (from dataset):**")
                st.markdown(f"- 🌡️ Temperature: {stats['Temperature']}")
                st.markdown(f"- 🌧️ Rainfall: {stats['Rainfall']}")
                st.markdown(f"- 🧪 pH: {stats['pH']}")
                st.markdown(f"- 💧 Humidity: {stats['Humidity']}")
                st.markdown("**Soil Requirements:**")
                st.markdown(f"- 🟢 Nitrogen (N): {stats['Nitrogen']}")
                st.markdown(f"- 🟣 Phosphorous (P): {stats['Phosphorous']}")
                st.markdown(f"- 🟠 Potassium (K): {stats['Potassium']}")
                
                st.caption(f"Similar to {main_crop.title()}")

def main():
    st.title("🌱 Smart Crop Recommendation System")
    st.markdown("Using your original dataset and trained model")
    
    # Sidebar with input parameters
    with st.sidebar:
        st.header("🌡️ Input Parameters")
        N = st.slider('Nitrogen (N) ratio in soil', 0, 150, 90)
        P = st.slider('Phosphorous (P) ratio in soil', 0, 150, 42)
        K = st.slider('Potassium (K) ratio in soil', 0, 150, 43)
        temperature = st.slider('Temperature (°C)', 0.0, 50.0, 20.88)
        humidity = st.slider('Humidity (%)', 0.0, 100.0, 82.0)
        ph = st.slider('pH value of soil', 0.0, 14.0, 6.5)
        rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 202.94)
        
        if st.button('🌱 Get Recommendation'):
            try:
                prediction = recommendation(N, P, K, temperature, humidity, ph, rainfall)
                crop_name = reverse_crop_dict.get(prediction, "Unknown Crop")
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
                st.error(f"❌ Error in prediction: {str(e)}")
        
        # Dataset information
        with st.expander("ℹ️ Dataset Info"):
            st.write(f"**Total Samples:** {len(crop_df)}")
            st.write(f"**Crop Varieties:** {len(crop_df['label'].unique())}")
            st.write("**Features:** N, P, K, temperature, humidity, ph, rainfall")

    # Main content area
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display recommended crop image
            display_crop_image(pred['crop_name'])
            
            # Display input parameters used
            st.subheader("📊 Input Parameters Used")
            input_df = pd.DataFrame.from_dict(pred['input_values'], orient='index', columns=['Value'])
            st.dataframe(input_df, use_container_width=True)
        
        with col2:
            # Display recommendation
            st.success(f"# 🌱 Recommended Crop: {pred['crop_name'].title()}")
            
            # Display actual stats from dataset
            st.subheader("🌱 Typical Growing Conditions")
            stats = get_crop_stats(pred['crop_name'])
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Temperature:** {stats['Temperature']}")
                st.markdown(f"**Rainfall:** {stats['Rainfall']}")
                st.markdown(f"**Nitrogen (N):** {stats['Nitrogen']}")
            with cols[1]:
                st.markdown(f"**pH Level:** {stats['pH']}")
                st.markdown(f"**Humidity:** {stats['Humidity']}")
                st.markdown(f"**Phosphorous (P):** {stats['Phosphorous']}")
                st.markdown(f"**Potassium (K):** {stats['Potassium']}")
            
            # Feature importance visualization
            st.subheader("📈 Feature Importance")
            features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            importance = rf.feature_importances_
            fig, ax = plt.subplots()
            sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
            ax.set_title('Most Important Factors for This Recommendation')
            st.pyplot(fig)
        
        # Display similar crops with actual dataset stats
        display_similar_crops(pred['similar_crops'], pred['crop_name'])
    else:
        st.info("ℹ️ Adjust the parameters in the sidebar and click 'Get Recommendation'")

    # Dataset statistics
    with st.expander("📚 Dataset Statistics"):
        st.write("### Nutrient Distribution Across Crops")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        sns.boxplot(data=crop_df, y='N', ax=ax[0])
        sns.boxplot(data=crop_df, y='P', ax=ax[1])
        sns.boxplot(data=crop_df, y='K', ax=ax[2])
        ax[0].set_title('Nitrogen (N)')
        ax[1].set_title('Phosphorous (P)')
        ax[2].set_title('Potassium (K)')
        st.pyplot(fig)

if __name__ == '__main__':
    main()

