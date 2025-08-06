import pandas as pd
import clip
import torch
import pickle
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_clip_features(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((200, 200))  
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image_input)
        return features.cpu().numpy().astype("float32")
    except Exception as e:
        print(f"[ERROR] Failed to extract image features: {e}")
        return None

def extract_text_features(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features.cpu().numpy().astype("float32")

def build_text_index(styles_csv_path="fashion_small/styles.csv", save_path="text_features.pkl"):
    print("⏳ Building text feature index from styles.csv...")
    df = pd.read_csv(styles_csv_path, on_bad_lines='skip')
    df['id'] = df['id'].astype(str) + ".jpg"

    id_list = []
    text_features = []

    for _, row in df.iterrows():
        if pd.isna(row['id']) or pd.isna(row['productDisplayName']):
            continue
        text = f"A photo of {row.get('gender', '')} {row.get('subCategory', '')} {row.get('productDisplayName', '')}"
        token = clip.tokenize([text]).to(device)
        with torch.no_grad():
            feat = model.encode_text(token).cpu().numpy().flatten()
        feat = feat / np.linalg.norm(feat) 
        text_features.append(feat)
        id_list.append(row['id'])

    text_features = np.array(text_features).astype("float32")

    with open(save_path, "wb") as f:
        pickle.dump((id_list, text_features), f)

    print("✅ Saved text feature index to", save_path)
