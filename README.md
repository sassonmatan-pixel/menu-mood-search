# Menu Mood Search

Semantic (vector) search over a restaurant menu, built with Streamlit and ChromaDB.
Describe a mood or craving ("something comforting and warm for a rainy day") and get the best-matching dishes
with photo, category, price and match confidence. Supports Hebrew (RTL) and English, with queries in either language.

## Run

    pip install -r requirements.txt
    streamlit run app.py

The first run downloads a multilingual embedding model (~120 MB) and builds the local ./menu_db Chroma store
(created automatically, not committed).
