import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import os
import numpy as np
import random
import faiss
import base64
from recommend import extract_clip_features, extract_text_features

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



st.set_page_config(page_title="AI Fashion Recommender", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .title-style {
            font-size: 48px;
            font-weight: 800;
            color: #ff4b4b;
            text-align: center;
            padding: 1rem;
            margin-bottom: 0.5rem;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }
        .caption-style {
            font-size: 18px;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .product-card {
            background-color: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
            transition: transform 0.3s ease;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #ff2b2b;
            transform: scale(1.02);
        }
        .uploaded-image-container {
            display: flex;
            justify-content: center;
            margin: 1rem 0;
            padding: 1rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .uploaded-image {
            max-width: 300px;
            max-height: 300px;
            border-radius: 8px;
            object-fit: contain;
        }
        .recommendation-title {
            font-size: 24px;
            font-weight: 700;
            color: #333;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #ff4b4b;
        }
        .price-tag {
            background-color: #ff4b4b;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: bold;
            display: inline-block;
            margin-top: 0.5rem;
        }
        
        .stRadio>div {
            flex-direction: row !important;
            gap: 1rem;
        }
        .stRadio>div>label {
            margin-bottom: 0;
        }
""", unsafe_allow_html=True)

st.markdown('<div class="title-style">👗 AI Fashion Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="caption-style">Discover your perfect style with AI-powered recommendations</div>', unsafe_allow_html=True)

styles_df = pd.read_csv("fashion_small/styles.csv", on_bad_lines='skip')
styles_df[styles_df.select_dtypes(include='object').columns] = styles_df.select_dtypes(include='object').fillna("Unknown")


brands = ["Zara", "H&M", "Nike", "Adidas", "Levi's", "Gucci", "Puma", "Louis Vuitton"]


metadata = {}
for _, row in styles_df.iterrows():
    img_name = str(row["id"]) + ".jpg"
    metadata[img_name] = {
        "productDisplayName": row.get("productDisplayName", "No Name"),
        "gender": row.get("gender", "Unisex"),
        "subCategory": row.get("subCategory", "Misc"),
        "brand": random.choice(brands),
        "price": random.randint(399, 4999)
    }
index = faiss.read_index("faiss_index.bin")
image_paths = pickle.load(open("image_paths.pkl", "rb"))
id_list, text_features = pickle.load(open("text_features.pkl", "rb"))


genders = sorted(list({v['gender'] for v in metadata.values()} - {""}))
categories = sorted(list({v['subCategory'] for v in metadata.values()} - {""}))

col1, col2 = st.columns([1, 2])
with col1:
    
    search_mode = st.radio("Search Mode", ["Image Upload", "Text Query"])
    selected_gender = st.selectbox("Filter by Gender", ["All"] + genders)
    selected_cat = st.selectbox("Filter by Category", ["All"] + categories)


def filter_results(paths):
    results = []
    for p in paths:
        fname = os.path.basename(p)
        meta = metadata.get(fname)
        if meta:
            if selected_gender != "All" and meta['gender'] != selected_gender:
                continue
            if selected_cat != "All" and meta['subCategory'] != selected_cat:
                continue
        results.append(p)
    return results[:6]

def show_recommendations(filtered_paths):
    st.markdown('<div class="recommendation-title">Recommended Products</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, path in enumerate(filtered_paths[:9]):
        with cols[i % 3]:
            meta = metadata.get(os.path.basename(path), {})
            st.image(path, use_container_width=True, 
                    caption=f"{meta.get('productDisplayName', '')}",
                    output_format="JPEG")
            
            st.markdown(f"**Brand**: {meta.get('brand', 'Unknown')}")
            st.markdown(f'<div class="price-tag">₹ {meta.get("price", "N/A")}</div>', 
                       unsafe_allow_html=True)


with col2:
    if search_mode == "Image Upload":
        uploaded = st.file_uploader("Upload an image of fashion item", type=["jpg", "png", "jpeg"], )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.markdown("""
            <div style='width: 500px; height: 500px; overflow: hidden; display: flex; justify-content: center; align-items: center; border-radius: 8px; margin: 10px 0;'>
                <img src='data:image/jpeg;base64,{}' style='max-height: 100%; max-width: 100%; object-fit: contain;'/>
            </div> 
            """.format(base64.b64encode(uploaded.getvalue()).decode()), 
            unsafe_allow_html=True)
            
            
            with open("query.jpg", "wb") as f:
                f.write(uploaded.getbuffer())
            query_vec = extract_clip_features("query.jpg")
            _, indices = index.search(query_vec, 20)
            top_paths = [image_paths[i] for i in indices[0]]
            filtered = filter_results(top_paths)
            show_recommendations(filtered)

    elif search_mode == "Text Query":
        user_query = st.text_input("Describe what you're looking for", 
                                  placeholder="e.g. women's summer dress, men's formal shirt")
        st.markdown("*Try these examples: 'women red handbag', 'men black sneakers', 'casual summer outfit'*")

        if user_query:
            prompt = f"A photo of {user_query.lower()}"
            query_vec = extract_text_features(prompt)
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True) 

            sims = np.dot(text_features, query_vec.T).squeeze()
            top_indices = sims.argsort()[-20:][::-1]
            top_paths = [os.path.join("fashion_small/images", id_list[i]) for i in top_indices]
    
            filtered = filter_results(top_paths)
            show_recommendations(filtered) 
            if not filtered:
                st.warning("No matching items found. Try a different query or adjust filters.")

st.markdown("---")
