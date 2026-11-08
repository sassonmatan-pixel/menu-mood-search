import base64
import html
import re
from pathlib import Path

import chromadb
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from menu_data import DIET_HE, DISHES, TEMP_HE, TEMP_TEXT, TRAIT_TEXT, TRAITS

DB_PATH = "./menu_db"
COLLECTION = "menu_e5_v4"
MODEL_REPO = "Xenova/multilingual-e5-small"
DIETS = ["Dairy", "Vegetarian", "Meat", "Vegan", "Fish"]
TEMPS = ["Hot", "Cold"]

TEXT = {
    "he": dict(
        dir="rtl", eyebrow="בחירת השף", title="מצאו את המנה המושלמת",
        sub="חיפוש וקטורי סמנטי מבין את <em>המשמעות</em> של התשוקה שלכם, ולא רק מילות מפתח. "
            "תארו מצב רוח, מזג אוויר או טעם, ואנחנו נתאים לכם מנה מהתפריט. אפשר לחפש בעברית או באנגלית.",
        label="מה בא לכם לאכול?", placeholder="לדוגמה: משהו חם ומנחם ליום גשום",
        count="מספר תוצאות", button="חיפוש", empty="נא להקליד משהו לחיפוש.",
        diet="סוג המנה", temp="טמפרטורה", any="הכול",
        results="{n} התאמות מובילות עבור “{q}”", match="רמת התאמה", none="לא נמצאו מנות שמתאימות לסינון.",
        loading="טוען את מסד נתוני התפריט (בהרצה הראשונה יורד מודל השפה, זה עלול לקחת זמן)...",
        menu_size="{n} מנות בתפריט"),
    "en": dict(
        dir="ltr", eyebrow="Chef's Selection", title="Find Your Perfect Dish",
        sub="Semantic vector search understands the <em>meaning</em> of your craving, not just keywords. "
            "Describe a mood, the weather or a flavour and we'll match it to the menu. You can search in English or Hebrew.",
        label="What are you in the mood for?", placeholder="e.g., something comforting and warm for a rainy day",
        count="Number of results", button="Search", empty="Please type something to search for.",
        diet="Dish type", temp="Temperature", any="All",
        results="Top {n} matches for “{q}”", match="Match confidence", none="No dishes match the selected filters.",
        loading="Loading menu database (first run downloads the language model)...",
        menu_size="{n} dishes on the menu"),
}


# ---------------------------------------------------------------- multilingual embeddings
class MultilingualE5:
    """multilingual-e5-small (ONNX): one vector space for Hebrew and English, tuned for query->passage search."""

    def __init__(self):
        import onnxruntime as ort
        model = hf_hub_download(MODEL_REPO, "onnx/model_quantized.onnx")
        tok = hf_hub_download(MODEL_REPO, "tokenizer.json")
        self.session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tok = Tokenizer.from_file(tok)
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding()

    def embed(self, texts, kind="passage"):
        enc = self.tok.encode_batch([f"{kind}: {t}" for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.session.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        pooled = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled.tolist()


@st.cache_resource
def get_engine():
    embedder = MultilingualE5()
    client = chromadb.PersistentClient(path=DB_PATH)
    for old in ("menu", "menu_multilingual", "menu_multilingual_v2", "menu_multilingual_v3"):  # earlier versions
        try:
            client.delete_collection(old)
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    # Idempotent: stable ids, loaded only if missing. Each dish is indexed twice (EN text, HE text)
    # so a query in either language matches cleanly; results are de-duplicated per dish.
    if col.count() < len(DISHES) * 2:
        ids, docs, metas = [], [], []
        for dish_id, _wiki, diet, temp, price, n_en, d_en, n_he, d_he in DISHES:
            meta = {"item": dish_id, "name_en": n_en, "name_he": n_he, "desc_en": d_en, "desc_he": d_he,
                    "category": diet, "diet": diet, "temp": temp, "price": float(price)}
            traits = TRAITS.get(dish_id, "").split()
            for i, (lg, name, desc) in enumerate((("en", n_en, d_en), ("he", n_he, d_he))):
                keywords = ", ".join([TEMP_TEXT[temp][i]] + [TRAIT_TEXT[t][i] for t in traits])
                ids.append(f"{dish_id}-{lg}")
                docs.append(f"{name}. {desc} {keywords}.")
                metas.append(meta)
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embedder.embed(docs))
    return col, embedder


STOP = {"something", "for", "the", "and", "with", "that", "from", "dish", "משהו", "מאוד", "עם", "של", "שיהיה", "לי", "אני"}


def norm_tokens(text: str) -> set:
    """Lower-cased word set; strips one leading Hebrew clitic letter (ה/ו/ב/ל/מ/כ/ש) from longer words."""
    out = set()
    for w in re.findall(r"[\w']+", text.lower()):
        if len(w) > 3 and w[0] in "והבלמכש" and "֐" <= w[0] <= "׿":
            out.add(w[1:])
        if len(w) >= 3 and w not in STOP:
            out.add(w)
    return out


def hybrid_hits(col, embedder, q, where, n):
    """Vector similarity + a small boost for keyword overlap (helps short Hebrew queries)."""
    total = col.count()
    res = col.query(query_embeddings=embedder.embed([q], "query"), n_results=total, where=where,
                    include=["metadatas", "distances", "documents"])
    q_tokens = norm_tokens(q)
    best = {}
    for meta, dist, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0]):
        overlap = len(q_tokens & norm_tokens(doc))
        sim = 1 - dist
        score = sim + 0.015 * min(overlap, 3)  # ranking score
        if meta["item"] not in best or score > best[meta["item"]][1]:
            best[meta["item"]] = (meta, score, sim)
    return sorted(best.values(), key=lambda x: -x[1])[:n]


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
.hero .count { display: inline-block; margin-top: 1rem; padding: .25rem .9rem; border-radius: 99px;
        background: rgba(255,255,255,.14); font-size: .82rem; color: #f6e6d6; }
[data-testid="stForm"] { background: #fff; border: none; border-radius: 18px; padding: 1.5rem 1.6rem;
        box-shadow: 0 10px 30px rgba(60,30,10,.10); margin-top: -4rem; position: relative; }
.stTextInput input { border-radius: 12px; padding: .8rem 1rem; font-size: 1rem; }
[data-testid="stFormSubmitButton"] button { background: #b4471e; color: #fff; border: none; border-radius: 12px;
        padding: .65rem 1.6rem; font-weight: 600; width: 100%; }
[data-testid="stFormSubmitButton"] button:hover { background: #8f3615; color: #fff; }
[data-testid="stButtonGroup"] button { border-radius: 10px; min-width: 2.6rem; }
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] { background: #b4471e; color: #fff; border-color: #b4471e; }
.results-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: #2b1a10; margin: 2rem 0 1rem; }
.card { display: flex; background: #fff; border-radius: 18px; overflow: hidden; margin-bottom: 1.1rem;
        box-shadow: 0 6px 22px rgba(60,30,10,.09); transition: transform .2s, box-shadow .2s; }
.card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(60,30,10,.16); }
.card img, .card .noimg { width: 230px; object-fit: cover; flex-shrink: 0; }
.card .noimg { background: linear-gradient(135deg, #e8b583, #b4471e); }
.card .body { padding: 1.2rem 1.4rem; flex: 1; display: flex; flex-direction: column; gap: .5rem; }
.card .top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.card h3 { font-family: 'Playfair Display', serif; font-size: 1.35rem; margin: 0; padding: 0; color: #2b1a10; }
.price { font-size: 1.35rem; font-weight: 700; color: #b4471e; white-space: nowrap; }
.tags { display: flex; gap: .4rem; flex-wrap: wrap; }
.tag { display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .72rem; font-weight: 600;
       letter-spacing: .06em; text-transform: uppercase; }
.desc { color: #5c4a3d; font-size: .93rem; line-height: 1.6; margin: 0; }
.match { margin-top: auto; }
.match .label { display: flex; justify-content: space-between; font-size: .78rem; color: #7a6657; margin-bottom: .3rem; }
.bar { height: 7px; background: #f0e6da; border-radius: 99px; overflow: hidden; }
.bar > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #e0933b, #b4471e); }
@media (max-width: 640px) { .card { flex-direction: column; } .card img, .card .noimg { width: 100%; height: 190px; }
        .hero { margin: 0 -1rem 2rem; padding: 2.5rem 1.2rem 5rem; } .hero h1 { font-size: 2.1rem; } }
"""
RTL_CSS = """
.stApp, .block-container, .stMarkdown, [data-testid="stForm"], .stTextInput label, .stRadio label,
[data-testid="stWidgetLabel"] { direction: rtl; text-align: right; }
.hero, .hero h1, .hero p { text-align: center; }
.hero h1, .results-title, .card h3 { font-family: 'Heebo', sans-serif; font-weight: 700; }
.hero .eyebrow, .tag { letter-spacing: normal; }
.stTextInput input { direction: rtl; text-align: right; }
"""
st.markdown(CSS + (RTL_CSS if RTL else "") + "</style>", unsafe_allow_html=True)

st.markdown(
    f'<div class="hero" dir="{T["dir"]}"><div class="eyebrow">{T["eyebrow"]}</div><h1>{T["title"]}</h1>'
    f'<p>{T["sub"]}</p><span class="count">{T["menu_size"].format(n=len(DISHES))}</span></div>',
    unsafe_allow_html=True,
)

TAG_COLORS = {"Dairy": ("#e3eefb", "#1f5f8f"), "Vegetarian": ("#e8f3d9", "#4b7a1d"), "Meat": ("#fde3dc", "#b3261e"),
              "Vegan": ("#e0f1dc", "#2e7d32"), "Fish": ("#dcf1f4", "#12708a"),
              "Hot": ("#ffe8cc", "#c25a00"), "Cold": ("#e4ecff", "#3b5bdb")}
TEMP_ICON = {"Hot": "🔥", "Cold": "❄️"}


@st.cache_data
def img_uri(item_id: str) -> str:
    p = Path(__file__).parent / "images" / f"{item_id}.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def diet_label(v: str) -> str:
    return DIET_HE[v] if RTL else v


def temp_label(v: str) -> str:
    return TEMP_HE[v] if RTL else v


with st.spinner(T["loading"]):
    collection, embedder = get_engine()

with st.form("search_form"):
    query = st.text_input(T["label"], placeholder=T["placeholder"])
    f1, f2 = st.columns(2)
    diets = f1.multiselect(T["diet"], DIETS, format_func=diet_label, placeholder=T["any"])
    temps = f2.multiselect(T["temp"], TEMPS, format_func=temp_label, placeholder=T["any"])
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    # Not st.slider: Streamlit's slider flips direction in browsers whose language is Hebrew (reversed bar/values)
    k = c1.segmented_control(T["count"], [1, 2, 3, 4, 5], default=3, selection_mode="single") or 3
    submitted = c2.form_submit_button(T["button"])

if submitted:
    st.session_state.last_search = (query.strip(), k, tuple(diets), tuple(temps)) if query.strip() else None
    st.session_state.empty = not query.strip()

if st.session_state.get("empty") and not st.session_state.get("last_search"):
    st.warning(T["empty"])

if st.session_state.get("last_search"):
    q, n, f_diets, f_temps = st.session_state.last_search
    conds = []
    if f_diets:
        conds.append({"diet": {"$in": list(f_diets)}})
    if f_temps:
        conds.append({"temp": {"$in": list(f_temps)}})
    where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
    hits = hybrid_hits(collection, embedder, q, where, n)
    if not hits:
        st.info(T["none"])
    else:
        st.markdown(f'<div class="results-title" dir="{T["dir"]}">{T["results"].format(n=len(hits), q=html.escape(q))}</div>',
                    unsafe_allow_html=True)
    for meta, _score, sim in hits:
        iid = meta["item"]
        conf = max(0.0, min(1.0, (sim - 0.74) / 0.14))  # e5 cosine scores sit in ~0.72-0.88
        name = html.escape(meta[f"name_{lang}"])
        desc = html.escape(meta[f"desc_{lang}"])
        tags = "".join(
            f'<span class="tag" style="background:{TAG_COLORS[v][0]};color:{TAG_COLORS[v][1]}">{icon}{html.escape(lbl)}</span>'
            for v, lbl, icon in ((meta["diet"], diet_label(meta["diet"]), ""),
                                 (meta["temp"], temp_label(meta["temp"]), TEMP_ICON[meta["temp"]] + " "))
        )
        uri = img_uri(iid)
        img = f'<img src="{uri}" alt="{name}">' if uri else '<div class="noimg"></div>'
        st.markdown(
            f'<div class="card" dir="{T["dir"]}">{img}'
            f'<div class="body"><div class="top"><h3>{name}</h3>'
            f'<div class="price"><bdi dir="ltr">${meta["price"]:.2f}</bdi></div></div>'
            f'<div class="tags">{tags}</div>'
            f'<p class="desc">{desc}</p>'
            f'<div class="match"><div class="label"><span>{T["match"]}</span><b><bdi dir="ltr">{conf:.0%}</bdi></b></div>'
            f'<div class="bar"><div style="width:{conf*100:.0f}%"></div></div></div></div></div>',
            unsafe_allow_html=True,
        )
