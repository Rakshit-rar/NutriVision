import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import torch
import open_clip
from engine import NutriSearch

st.set_page_config(page_title='NutriVision AI', layout='wide')

@st.cache_resource
def load_production_assets():
    # Load CLIP for real food classification
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    nutri = NutriSearch('nutrition.csv')
    return model, preprocess, tokenizer, nutri

model, preprocess, tokenizer, nutri = load_production_assets()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

st.title('☕ NutriVision AI Platform')
st.markdown('### Intelligent Food Recognition & Nutrition Dashboard')

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Food Image")
    uploaded_file = st.file_uploader("Select a photo...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

with col2:
    st.header("AI Analysis")
    if uploaded_file and st.button('Identify & Analyze'):
        with st.spinner('AI is identifying the dish...'):
            # 1. Get candidate labels from database
            labels = nutri.df['food_name'].unique().tolist()
            
            # 2. CLIP Zero-Shot Inference
            image_input = preprocess(img).unsqueeze(0).to(device)
            text_tokens = tokenizer([f'a photo of {l}' for l in labels]).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                best_match_idx = similarity.argmax().item()
            
            detected_food = labels[best_match_idx]
            data = nutri.get_nutrition_by_name(detected_food)
            
            if "error" not in data:
                st.metric("Detected Item", detected_food.title())
                
                # Macros Pie Chart
                df_viz = pd.DataFrame({
                    'Nutrient': ['Protein', 'Carbs', 'Fat'],
                    'Grams': [data['protein'], data['carbs'], data['fat']]
                })
                fig = px.pie(df_viz, values='Grams', names='Nutrient', hole=0.4, title='Nutritional Macros')
                st.plotly_chart(fig)
                
                st.success(f"Estimated Calories: {data['calories']} kcal")
                st.caption("Nutritional values are estimated per 100g serving.")
            else:
                st.error("Item identified but nutritional profile is missing from database.")
