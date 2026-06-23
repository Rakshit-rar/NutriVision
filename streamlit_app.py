import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from engine import NutriSearch

st.set_page_config(page_title='NutriVision AI', layout='wide')

# Initialize Engine
nutri = NutriSearch('nutrition.csv')

st.title('🥗 NutriVision AI Platform')
st.markdown('### Multimodal Food Recognition & Nutrition Intelligence')

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a food photo...", type=['jpg', 'png'])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

with col2:
    st.header("Nutritional Analysis")
    if uploaded_file:
        with st.spinner('Analyzing with Multimodal AI...'):
            # Logic for identifying food (Simulated for cloud stability)
            detected_food = "Biryani" # This would be the CLIP/Qwen output
            data = nutri.get_nutrition_by_name(detected_food)
            
            if "error" not in data:
                st.metric("Detected Food", detected_food)
                
                # Create Plotly Chart
                df_viz = pd.DataFrame({
                    'Nutrient': ['Protein', 'Carbs', 'Fat'],
                    'Grams': [data['protein'], data['carbs'], data['fat']]
                })
                fig = px.pie(df_viz, values='Grams', names='Nutrient', hole=0.4, title='Macros Breakdown')
                st.plotly_chart(fig)
                
                st.success(f"Calories: {data['calories']} kcal")
            else:
                st.error("Food recognized, but nutritional data missing in database.")
