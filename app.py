import base64
import html
from pathlib import Path

import chromadb
import numpy as np
import streamlit as st
from chromadb import Documents, EmbeddingFunction, Embeddings
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DB_PATH = "./menu_db"
COLLECTION = "menu_multilingual_v2"
MODEL_REPO = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------- menu data (EN + HE)
MENU = [
    dict(en="Diavolo Inferno Ramen", he="ראמן דיאבולו אינפרנו", cat="Spicy", price=16.50,
         d_en="Fiery noodle soup in a chili-oil pork broth with ghost pepper paste, soft egg, scallions and a slow-building heat that makes your forehead sweat.",
         d_he="מרק אטריות להבה במרק בשר חזיר בשמן צ'ילי עם משחת פלפל רפאים, ביצה רכה ובצל ירוק, עם חריפות שמתעצמת לאט ומעלה זיעה במצח."),
    dict(en="Szechuan Fire Wings", he="כנפי אש סצ'ואן", cat="Spicy", price=13.00,
         d_en="Crispy chicken wings tossed in numbing Szechuan peppercorn, dried red chilies and garlic; bold, tingly and intensely hot.",
         d_he="כנפי עוף פריכות מוקפצות בפלפל סצ'ואן מרדים, צ'ילי אדום יבש ושום; טעם נועז, מעקצץ וחריף במיוחד."),
    dict(en="Garden Buddha Bowl", he="קערת בודהה גן", cat="Vegan", price=14.25,
         d_en="Light, fresh bowl of quinoa, roasted chickpeas, avocado, shredded kale and carrot with a lemon-tahini dressing. Plant-based and healthy.",
         d_he="קערה קלילה וטרייה של קינואה, חומוס קלוי, אבוקדו, קייל קצוץ וגזר ברוטב לימון וטחינה. טבעוני ובריא."),
    dict(en="Coconut Lentil Dal", he="דאל עדשים וקוקוס", cat="Vegan", price=12.75,
         d_en="Creamy red lentils simmered with coconut milk, turmeric, ginger and cumin; a comforting, warm, plant-based stew served with flatbread.",
         d_he="עדשים כתומות קרמיות מבושלות בחלב קוקוס, כורכום, ג'ינג'ר וכמון; תבשיל טבעוני חם ומנחם המוגש עם לחם שטוח."),
    dict(en="Slow-Braised Beef Stew", he="נזיד בקר בבישול איטי", cat="Hearty", price=19.00,
         d_en="Tender beef shoulder braised for hours with root vegetables, red wine and thyme; rich, warming comfort food for a cold rainy day.",
         d_he="כתף בקר רכה שבושלה שעות עם ירקות שורש, יין אדום וטימין; אוכל מנחם, עשיר ומחמם ליום גשום וקר."),
    dict(en="Loaded Mac & Cheese", he="מקרוני וגבינה עמוס", cat="Hearty", price=15.50,
         d_en="Baked cavatappi in a three-cheese sauce with crispy bacon and a golden breadcrumb crust; gooey, filling and indulgent.",
         d_he="פסטה קוואטאפי אפויה ברוטב שלוש גבינות עם בייקון פריך ופירורי לחם מוזהבים; מתיחה, משביעה ומפנקת."),
    dict(en="Molten Chocolate Lava Cake", he="עוגת לבה שוקולד חמה", cat="Dessert", price=9.50,
         d_en="Warm dark chocolate cake with a gooey molten center, served with vanilla bean ice cream. Decadent and sweet.",
         d_he="עוגת שוקולד מריר חמה עם מרכז נוזלי, מוגשת עם גלידת וניל. מפנקת ומתוקה."),
    dict(en="Lemon Berry Sorbet", he="סורבה לימון ופירות יער", cat="Dessert", price=7.25,
         d_en="Refreshing, dairy-free sorbet of tangy lemon and wild berries; light, cold and zesty palate cleanser.",
         d_he="סורבה מרענן וללא חלב מלימון חמצמץ ופירות יער; קליל, קר ומרענן את החך."),
    dict(en="Classic Chicken Noodle Soup", he="מרק עוף עם אטריות קלאסי", cat="Soup", price=10.75,
         d_en="Homestyle broth with tender chicken, egg noodles, carrots and celery; a soothing, warm bowl that feels like a hug when you're under the weather.",
         d_he="מרק ביתי עם עוף רך, אטריות ביצים, גזר וסלרי; קערה חמה ומרגיעה שמרגישה כמו חיבוק כשאתם לא מרגישים טוב."),
    dict(en="Grilled Salmon with Asparagus", he="סלמון על הגריל עם אספרגוס", cat="Seafood", price=22.00,
         d_en="Flaky grilled Atlantic salmon with charred asparagus, herb butter and lemon; light, elegant and protein-rich.",
         d_he="סלמון אטלנטי צלוי על הגריל עם אספרגוס חרוך, חמאת עשבי תיבול ולימון; קליל, אלגנטי ועשיר בחלבון."),
]

CATEGORY_HE = {"Spicy": "חריף", "Vegan": "טבעוני", "Hearty": "משביע", "Dessert": "קינוח",
               "Soup": "מרק", "Seafood": "פירות ים"}

TEXT = {
    "he": dict(
        dir="rtl", eyebrow="בחירת השף", title="מצאו את המנה המושלמת",
        sub="חיפוש וקטורי סמנטי מבין את <em>המשמעות</em> של התשוקה שלכם, ולא רק מילות מפתח. "
            "תארו מצב רוח, מזג אוויר או טעם, ואנחנו נתאים לכם מנה מהתפריט. אפשר לחפש בעברית או באנגלית.",
        label="מה בא לכם לאכול?", placeholder="לדוגמה: משהו חם ומנחם ליום גשום",
        count="מספר תוצאות", button="חיפוש", empty="נא להקליד משהו לחיפוש.",
        results="{n} התאמות מובילות עבור “{q}”", match="רמת התאמה",
        loading="טוען את מסד נתוני התפריט (בהרצה הראשונה יורד מודל השפה, זה עלול לקחת זמן)...",
        lang="שפה", example="נסו לחפש:"),
    "en": dict(
        dir="ltr", eyebrow="Chef's Selection", title="Find Your Perfect Dish",
        sub="Semantic vector search understands the <em>meaning</em> of your craving, not just keywords. "
            "Describe a mood, the weather or a flavour and we'll match it to the menu. You can search in English or Hebrew.",
        label="What are you in the mood for?", placeholder="e.g., something comforting and warm for a rainy day",
        count="Number of results", button="Search", empty="Please type something to search for.",
        results="Top {n} matches for “{q}”", match="Match confidence",
        loading="Loading menu database (first run downloads the language model)...",
        lang="Language", example="Try:"),
}


# ---------------------------------------------------------------- multilingual embeddings
class MultilingualONNX(EmbeddingFunction[Documents]):
    """paraphrase-multilingual-MiniLM-L12-v2 (ONNX): same vector space for Hebrew and English."""

    def __init__(self):
        import onnxruntime as ort
        model = hf_hub_download(MODEL_REPO, "onnx/model_quantized.onnx")
        tok = hf_hub_download(MODEL_REPO, "tokenizer.json")
        self.session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(tok)
        self.tok.enable_truncation(max_length=128)
        self.tok.enable_padding()

    def __call__(self, input: Documents) -> Embeddings:
        enc = self.tok.encode_batch(list(input))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        out = self.session.run(None, {"input_ids": ids, "attention_mask": mask,
                                      "token_type_ids": np.zeros_like(ids)})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
        return [v for v in pooled]

    @staticmethod
    def name() -> str:
        return "multilingual-minilm-onnx"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return MultilingualONNX()


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    for old in ("menu", "menu_multilingual"):  # remove collections from earlier versions
        try:
            client.delete_collection(old)
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION, embedding_function=MultilingualONNX(),
                                          metadata={"hnsw:space": "cosine"})
    # Idempotent: stable ids, loaded only if missing. Each dish is indexed twice (EN text, HE text)
    # so a query in either language matches cleanly; results are de-duplicated per dish.
    if col.count() < len(MENU) * 2:
        ids, docs, metas = [], [], []
        for i, m in enumerate(MENU):
            meta = {"item": f"item-{i}", "name_en": m["en"], "name_he": m["he"], "desc_en": m["d_en"],
                    "desc_he": m["d_he"], "category": m["cat"], "price": float(m["price"])}
            for lg in ("en", "he"):
                ids.append(f"item-{i}-{lg}")
                docs.append(f"{m[lg]}. {m['d_' + lg]}")
                metas.append(meta)
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    return col


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Semantic Menu Search | חיפוש תפריט חכם", page_icon="🍽️", layout="centered")

if "lang" not in st.session_state:
    st.session_state.lang = "he"

top = st.columns([3, 2])
choice = top[1].radio("Language / שפה", ["עברית", "English"], horizontal=True,
                      index=0 if st.session_state.lang == "he" else 1, label_visibility="collapsed")
lang = "he" if choice == "עברית" else "en"
st.session_state.lang = lang
T = TEXT[lang]
RTL = lang == "he"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Heebo:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Heebo', sans-serif; }
.stApp { background: #faf6f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 860px; padding-top: 1rem !important; }
.hero { background: linear-gradient(135deg, rgba(30,20,12,.84), rgba(120,50,20,.74)),
        url('https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=1400') center/cover, #3b2314;
        margin: 0 -4rem 2rem; padding: 3.5rem 4rem 5rem; text-align: center; color: #fff;
        border-radius: 28px; }
.hero h1 { font-family: 'Playfair Display', serif; font-size: 3rem; margin: 0; padding: 0; color: #fff; }
.hero p { max-width: 580px; margin: .8rem auto 0; color: #f1dfcf; font-size: 1.02rem; line-height: 1.6; }
.hero .eyebrow { letter-spacing: .25em; text-transform: uppercase; font-size: .75rem; color: #e8b583; margin-bottom: .6rem; }
[data-testid="stForm"] { background: #fff; border: none; border-radius: 18px; padding: 1.5rem 1.6rem;
        box-shadow: 0 10px 30px rgba(60,30,10,.10); margin-top: -4rem; position: relative; }
.stTextInput input { border-radius: 12px; padding: .8rem 1rem; font-size: 1rem; }
[data-testid="stFormSubmitButton"] button { background: #b4471e; color: #fff; border: none; border-radius: 12px;
        padding: .65rem 1.6rem; font-weight: 600; width: 100%; }
[data-testid="stFormSubmitButton"] button:hover { background: #8f3615; color: #fff; }
.results-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: #2b1a10; margin: 2rem 0 1rem; }
.card { display: flex; background: #fff; border-radius: 18px; overflow: hidden; margin-bottom: 1.1rem;
        box-shadow: 0 6px 22px rgba(60,30,10,.09); transition: transform .2s, box-shadow .2s; }
.card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(60,30,10,.16); }
.card img { width: 230px; object-fit: cover; flex-shrink: 0; }
.card .body { padding: 1.2rem 1.4rem; flex: 1; display: flex; flex-direction: column; gap: .5rem; }
.card .top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.card h3 { font-family: 'Playfair Display', serif; font-size: 1.35rem; margin: 0; padding: 0; color: #2b1a10; }
.price { font-size: 1.35rem; font-weight: 700; color: #b4471e; white-space: nowrap; }
.tag { display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .72rem; font-weight: 600;
       letter-spacing: .06em; text-transform: uppercase; align-self: flex-start; }
.desc { color: #5c4a3d; font-size: .93rem; line-height: 1.6; margin: 0; }
.match { margin-top: auto; }
.match .label { display: flex; justify-content: space-between; font-size: .78rem; color: #7a6657; margin-bottom: .3rem; }
.bar { height: 7px; background: #f0e6da; border-radius: 99px; overflow: hidden; }
.bar > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #e0933b, #b4471e); }
@media (max-width: 640px) { .card { flex-direction: column; } .card img { width: 100%; height: 190px; }
        .hero { margin: 0 -1rem 2rem; padding: 2.5rem 1.2rem 5rem; } .hero h1 { font-size: 2.1rem; } }
"""
RTL_CSS = """
.stApp, .block-container, .stMarkdown, [data-testid="stForm"], .stTextInput label, .stSlider label,
.stRadio label, [data-testid="stWidgetLabel"] { direction: rtl; text-align: right; }
.hero, .hero h1, .hero p { text-align: center; }
.hero h1, .results-title, .card h3 { font-family: 'Heebo', sans-serif; font-weight: 700; }
.hero .eyebrow { letter-spacing: normal; }
.tag { letter-spacing: normal; }
.stTextInput input { direction: rtl; text-align: right; }
[data-testid="stSlider"] > div:nth-child(2) { direction: ltr; }
"""
st.markdown(CSS + (RTL_CSS if RTL else "") + "</style>", unsafe_allow_html=True)

st.markdown(
    f'<div class="hero" dir="{T["dir"]}"><div class="eyebrow">{T["eyebrow"]}</div><h1>{T["title"]}</h1>'
    f'<p>{T["sub"]}</p></div>',
    unsafe_allow_html=True,
)

TAG_COLORS = {"Spicy": ("#fde3dc", "#b3261e"), "Vegan": ("#e0f1dc", "#2e7d32"), "Hearty": ("#f5e6d3", "#8a5a1f"),
              "Dessert": ("#fbe1ee", "#a8326f"), "Soup": ("#fff0cc", "#946200"), "Seafood": ("#dcecf8", "#1f5f8f")}


@st.cache_data
def img_uri(item_id: str) -> str:
    p = Path(__file__).parent / "images" / f"{item_id}.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


with st.spinner(T["loading"]):
    collection = get_collection()

with st.form("search_form"):
    query = st.text_input(T["label"], placeholder=T["placeholder"])
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    k = c1.slider(T["count"], 1, 5, 3)
    submitted = c2.form_submit_button(T["button"])

if submitted:
    st.session_state.last_search = (query.strip(), k) if query.strip() else None
    st.session_state.empty = not query.strip()

if st.session_state.get("empty") and not st.session_state.get("last_search"):
    st.warning(T["empty"])

if st.session_state.get("last_search"):
    q, n = st.session_state.last_search
    res = collection.query(query_texts=[q], n_results=min(n * 2, collection.count()))
    seen, hits = set(), []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):  # best-first; keep best language per dish
        if meta["item"] not in seen:
            seen.add(meta["item"])
            hits.append((meta, dist))
    hits = hits[:n]
    st.markdown(f'<div class="results-title" dir="{T["dir"]}">{T["results"].format(n=len(hits), q=html.escape(q))}</div>',
                unsafe_allow_html=True)
    for meta, dist in hits:
        iid = meta["item"]
        conf = max(0.0, min(1.0, 1 - dist))
        bg, fg = TAG_COLORS.get(meta["category"], ("#eee", "#333"))
        name = html.escape(meta[f"name_{lang}"])
        desc = html.escape(meta[f"desc_{lang}"])
        cat = html.escape(CATEGORY_HE[meta["category"]] if RTL else meta["category"])
        st.markdown(
            f'<div class="card" dir="{T["dir"]}"><img src="{img_uri(iid)}" alt="{name}">'
            f'<div class="body"><div class="top"><h3>{name}</h3>'
            f'<div class="price"><bdi dir="ltr">${meta["price"]:.2f}</bdi></div></div>'
            f'<span class="tag" style="background:{bg};color:{fg}">{cat}</span>'
            f'<p class="desc">{desc}</p>'
            f'<div class="match"><div class="label"><span>{T["match"]}</span><b><bdi dir="ltr">{conf:.0%}</bdi></b></div>'
            f'<div class="bar"><div style="width:{conf*100:.0f}%"></div></div></div></div></div>',
            unsafe_allow_html=True,
        )
