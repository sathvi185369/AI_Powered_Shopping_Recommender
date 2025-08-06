import os
import clip
import torch
import pandas as pd
from PIL import Image
import numpy as np
import faiss
import pickle
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


df = pd.read_csv("fashion_small/styles.csv", on_bad_lines='skip')
df['id'] = df['id'].astype(str) + ".jpg"
valid_ids = set(df['id'])

image_folder = "fashion_small/images"
image_paths = []
features = []

print("Extracting image features...")
for fname in tqdm(os.listdir(image_folder)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")) and fname in valid_ids:
        fpath = os.path.join(image_folder, fname)
        try:
            img = preprocess(Image.open(fpath).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(img).cpu().numpy().flatten()
            features.append(feature)
            image_paths.append(fpath)
        except Exception as e:
            print(f"Error: {fname} => {e}")

features = np.array(features).astype("float32")
features = features / np.linalg.norm(features, axis=1, keepdims=True)


faiss_index = faiss.IndexFlatL2(features.shape[1])
faiss_index.add(features)
faiss.write_index(faiss_index, "faiss_index.bin")


with open("image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)


id_to_meta = {}
for _, row in df.iterrows():
    id_to_meta[row['id']] = {
        "gender": row.get("gender", ""),
        "masterCategory": row.get("masterCategory", ""),
        "subCategory": row.get("subCategory", ""),
        "productDisplayName": row.get("productDisplayName", "")
    }

with open("metadata_map.pkl", "wb") as f:
    pickle.dump(id_to_meta, f)

print("✅ Feature extraction complete. Index and metadata saved.")
