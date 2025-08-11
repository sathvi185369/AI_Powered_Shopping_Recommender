

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/fc107ca7-2c87-4c1d-a4fc-b29012c881ec" />

An intelligent, multi-modal shopping recommendation system built using OpenAI’s CLIP for feature extraction and FAISS for fast similarity search.
Users can search for products by text or image, and the system returns visually and semantically similar items from the product catalog.

# Features

🔍 Multi-modal Search — Search using text queries or uploaded images.

🧠 CLIP-based Embeddings — Extracts rich visual and semantic features.

⚡ FAISS Vector Search — Lightning-fast similarity search.

🏷 Brand & Category Filtering — View results by brand or category.

📈 Scalable & Efficient — Handles large datasets.

🌐 Django Backend — Easy to integrate into e-commerce sites.

🎨 Streamlit / HTML UI — Simple and interactive interface.

<img width="1903" height="710" alt="Screenshot 2025-08-11 153058" src="https://github.com/user-attachments/assets/fbaf0593-fd56-4d11-a9a7-a255e2c60f3b" />


# System Architecture

🛠 Tech Stack

Component               	Technology

Backend	                  Django / Python
Deep Learning	            OpenAI CLIP (PyTorch)
Vector Search	            FAISS
Frontend	                HTML, CSS, JS / Streamlit
Data Handling	            Pandas, NumPy
Image Processing	        Pillow, OpenCV




⚙️ Installation

1️⃣ Clone the repository

git clone https://github.com/yourusername/ai-shopping-recommender.git
cd ai-shopping-recommender

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Prepare dataset

Store all product images in data/images/

Include styles.csv with:
id, category, brand, gender, color, productDisplayName

4️⃣ Build FAISS index

# Encode input (text or image) → Search FAISS → Retrieve top-N matches
query_features = encode_with_clip(query)  
results = faiss_index.search(query_features, k=5)  
display_results(results)

<img width="1215" height="993" alt="Screenshot 2025-08-11 153311" src="https://github.com/user-attachments/assets/d0f79a57-6e04-485a-bde6-e25ab79cddc2" />


# How It Works

-Preprocessing

-CLIP extracts image & text embeddings

-Features are normalized & stored in FAISS

-Searching

-Encode query (image/text) with CLIP

-Search embeddings in FAISS index

-Display top matches with details


# Future Enhancements

-Price-based filtering

-Live product API integration

-Multi-language search

-Dockerized deployment

