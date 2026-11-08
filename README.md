# Menu Mood Search

Semantic (vector) search over a 61-dish restaurant menu, built with Streamlit and ChromaDB.
Describe a mood or craving ("something comforting and warm for a rainy day") and get the best-matching dishes
with photo, tags, price and match confidence.

- Hebrew (right-to-left) and English UI, with queries in either language
- Filters by dish type (dairy, vegetarian, meat, vegan, fish) and temperature (hot / cold)
- Embeddings: multilingual-e5-small (ONNX) + a small keyword boost; vector store: ChromaDB (persistent, `./menu_db`)

## Run

    pip install -r requirements.txt
    streamlit run app.py

The first run downloads the embedding model (~120 MB) and builds the local `./menu_db` store (not committed).

## Files

- `app.py` - the Streamlit app
- `menu_data.py` - all dishes (EN + HE), search keywords
- `fetch_images.py` - downloads dish photos from Wikipedia into `images/` (already included)
