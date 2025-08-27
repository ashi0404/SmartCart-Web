import os, io, requests
import streamlit as st
import pandas as pd
from collections import Counter
from typing import Optional

from data_loader import (
    build_items_and_tags, build_normalized_comatrix,
    save_artifact, load_artifact, extract_item_names, clean_item_list
)
from recommender import enhanced_recommend, batch_predict, normalize_user_items
from ui_components import (
    icon_for_item, TYPE_EMOJI, topbar_badges, reco_card, header
)

APP_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
ART_DIR = os.path.join(BASE_DIR, "artifacts")
ASSET_DIR = os.path.join(APP_DIR, "assets")
LOGO_PATH = os.path.join(ASSET_DIR, "logo.png")  # optional

# ====== About page constants ======
APP_BRAND_LINE = (
    "We are **Team JaiMataDi**.\n\n"
    "We built **SmartCart** — a lightweight recommendation engine that turns static upsells "
    "into personalized suggestions across apps, web, and kiosks.\n\n"
    "In our tests, SmartCart improved **Recall@3** and **Precision@3** by **150%** over the baseline. "
    "That means customers see more relevant add-ons — and **Wings R Us** captures more revenue per order."
)

TEAM = [
    {"name": "Bhaskar Ranjan Karn","linkedin": "https://www.linkedin.com/in/bhaskar-ranjan-karn/","photo": os.path.join(BASE_DIR, "assets", "team", "bhaskar.jpg")},
    {"name": "Astitva","linkedin": "https://www.linkedin.com/in/astitva-07a338229/","photo": os.path.join(BASE_DIR, "assets", "team", "astitva.jpg")},
    {"name": "Sanjay Kumar","linkedin": "https://www.linkedin.com/in/sanjay-kumar-39b73a239/","photo": os.path.join(BASE_DIR, "assets", "team", "sanjay.jpg")},
]

st.set_page_config(page_title="SmartCart-Web — Menu Recommender", page_icon="🍗", layout="wide")

# Load CSS
with open(os.path.join(APP_DIR, "styles.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.markdown(
    """
    <div class="brand-side">
      <span class="brand-blue">SmartCart</span><span class="brand-red">-Web</span>
    </div>
    """,
    unsafe_allow_html=True
)
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, caption="", use_container_width=True)

page = st.sidebar.radio(
    "",
    ["🏁 Start","🧱 Build Model (first run)","🛒 Menu & Recommendations",
     "📦 Batch Predict (CSV)","📊 Metrics & Explore","🧩 Architecture & Workflow","ℹ️ About"],
    label_visibility="collapsed"
)

# ---------- HELPERS ----------
def app_brand_title():
    st.markdown(
        """
        <h1 class="brand-center">
          <span class="brand-blue">SmartCart</span><span class="brand-red">-Web</span>
        </h1>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def save_uploaded_file(uploaded_file, filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, filename)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path

def download_from_gdrive(url: str, filename: str):
    """Download CSV from Google Drive link and save to DATA_DIR."""
    try:
        file_id = url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(download_url)
        r.raise_for_status()
        os.makedirs(DATA_DIR, exist_ok=True)
        out_path = os.path.join(DATA_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    except Exception as e:
        st.error(f"Failed to fetch from Google Drive: {e}")
        return None

@st.cache_data(show_spinner=True)
def prepare_artifacts(sample_n: Optional[int]):
    order_path = os.path.join(DATA_DIR, "order_data.csv")
    if not os.path.exists(order_path):
        st.error("Please upload order_data.csv first in Start page.")
        return None
    order = pd.read_csv(order_path, chunksize=200000)  # ✅ large-file safe
    order = pd.concat(order, ignore_index=True)

    order["ITEM_LIST"] = order["ORDERS"].apply(extract_item_names).apply(clean_item_list)
    item_type, item_feat, top_by_type, all_items = build_items_and_tags(order)
    co_norm = build_normalized_comatrix(order, sample_n=sample_n)

    known_lower = {itm.lower(): itm for itm in all_items}
    art = {"item_type": item_type,"item_feat": item_feat,"top_by_type": top_by_type,
           "co_norm": co_norm,"known_items_lower": list(known_lower.keys()),"lower_to_orig": known_lower}
    save_artifact("artifacts.pkl", art)
    return art

def load_or_build_artifacts():
    art = load_artifact("artifacts.pkl")
    if art is None:
        st.warning("Artifacts not found. Please go to **Build Model (first run)**.")
        return None
    return art

# ---------- PAGES ----------
def start_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Welcome</h2>", unsafe_allow_html=True)
    st.write("Upload required CSV files or paste Google Drive links (large files >700MB are supported).")

    # Uploaders
    uploaded_order = st.file_uploader("Upload order_data.csv", type="csv")
    gdrive_order = st.text_input("Or paste Google Drive link for order_data.csv")

    uploaded_customer = st.file_uploader("Upload customer_data.csv", type="csv")
    gdrive_customer = st.text_input("Or paste Google Drive link for customer_data.csv")

    uploaded_store = st.file_uploader("Upload store_data.csv", type="csv")
    gdrive_store = st.text_input("Or paste Google Drive link for store_data.csv")

    uploaded_test = st.file_uploader("Upload test_data_question.csv", type="csv")
    gdrive_test = st.text_input("Or paste Google Drive link for test_data_question.csv")

    # Save logic
    if uploaded_order: save_uploaded_file(uploaded_order, "order_data.csv")
    elif gdrive_order: download_from_gdrive(gdrive_order, "order_data.csv")

    if uploaded_customer: save_uploaded_file(uploaded_customer, "customer_data.csv")
    elif gdrive_customer: download_from_gdrive(gdrive_customer, "customer_data.csv")

    if uploaded_store: save_uploaded_file(uploaded_store, "store_data.csv")
    elif gdrive_store: download_from_gdrive(gdrive_store, "store_data.csv")

    if uploaded_test: save_uploaded_file(uploaded_test, "test_data_question.csv")
    elif gdrive_test: download_from_gdrive(gdrive_test, "test_data_question.csv")

    if all([os.path.exists(os.path.join(DATA_DIR, f)) for f in
            ["order_data.csv","customer_data.csv","store_data.csv","test_data_question.csv"]]):
        st.success("✅ All files ready under data/")

def build_model_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Build Model (first run)</h2>", unsafe_allow_html=True)
    sample = st.slider("Sample N orders (speed vs quality)",100_000, 1_414_410, 250_000, step=50_000)
    if st.button("🚀 Build now"):
        with st.spinner("Crunching pairs & normalizing..."):
            art = prepare_artifacts(sample_n=sample)
        if art: st.success("Artifacts built and cached at artifacts/artifacts.pkl")

def menu_reco_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Menu & Recommendations</h2>", unsafe_allow_html=True)
    art = load_or_build_artifacts()
    if art is None: return

    all_items = list(art["lower_to_orig"].values())
    selected = st.multiselect("Select up to 3 items", options=all_items, default=[])
    if len(selected) > 3:
        st.error("Only 3 items allowed.")
        selected = selected[:3]
    topbar_badges(selected, limit=3)

    if st.button("🍽️ Recommend", disabled=(len(selected) == 0)):
        cart = normalize_user_items(selected, art["known_items_lower"], art["lower_to_orig"])
        recs = enhanced_recommend(cart, art["co_norm"], art["item_type"], art["top_by_type"], art["item_feat"])
        if not recs: st.warning("No recommendations found."); return
        col_a, col_b, col_c = st.columns(3)
        for idx, (it, score) in enumerate(recs, start=1):
            with [col_a, col_b, col_c][idx - 1]:
                reco_card(idx, it, score, art["item_type"].get(it, "other"))

def batch_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Batch Predict (CSV)</h2>", unsafe_allow_html=True)
    art = load_or_build_artifacts()
    if art is None: return

    test_path = os.path.join(DATA_DIR, "test_data_question.csv")
    if not os.path.exists(test_path):
        st.error("Please upload test_data_question.csv in Start page."); return

    if st.button("Run batch on test_data_question.csv"):
        test_df = pd.read_csv(test_path)
        out = batch_predict(test_df, art["co_norm"], art["item_type"], art["top_by_type"], art["item_feat"],
                            art["known_items_lower"], art["lower_to_orig"])
        out_path = os.path.join(ART_DIR, "SmartCart_Recommendation_Output.csv")
        out.to_csv(out_path, index=False)
        st.success(f"Saved: {out_path}")
        st.dataframe(out.head(20))
        with open(out_path, "rb") as f:
            st.download_button("Download CSV", f, file_name="SmartCart_Recommendation_Output.csv", mime="text/csv")

def metrics_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Metrics & Explore</h2>", unsafe_allow_html=True)
    art = load_or_build_artifacts()
    if art is None: return

    counts = Counter(art["item_type"].values())
    st.bar_chart(pd.DataFrame.from_dict(counts, orient="index", columns=["Count"]))

    st.markdown("<h4 class='page-h4'>Top Items by Type</h4>", unsafe_allow_html=True)
    cols = st.columns(4)
    for t, col in zip(["main", "side", "dip", "drink"], cols):
        with col:
            st.markdown(f"**{t.title()}** {TYPE_EMOJI.get(t, '🍽️')}")
            for it, cnt in art["top_by_type"].get(t, [])[:15]:
                st.write(f"{icon_for_item(it)} {it} — {cnt}")

def render_workflow_diagram():
    mermaid_code = """flowchart TD
    subgraph Preprocessing["Data Preprocessing"]
        A["Raw Data (Orders, Customers, Stores)"]
        B["Data Cleaning (Parse JSON, Remove Noise, Handle Nulls)"]
        C["Item Tagging (Main, Side, Drink, Dip, Dessert + Veg/Spicy/Combo)"]
    end
    subgraph FeatureEng["Co-occurrence Matrix"]
        D["Build Matrix (Pair Counts + Normalization)"]
    end
    subgraph Scoring["Scoring + Filters"]
        E1["Base Scoring (Highest Co-occurrence Items)"]
        E2["Filters (Blacklist Removal + Normalization)"]
        F["Soft Bias (Boost Missing Categories)"]
    end
    subgraph Output["Recommendations + Evaluation"]
        G["Smart Recommendations (Top-3 Items with Fallback Variety)"]
        H["Evaluation (Recall@3, Precision@3, Top-1 Accuracy)"]
        I["Business Output (Excel with 3 Recos/Test Cart)"]
    end
    A --> B --> C --> D
    D --> E1 --> F
    D --> E2 --> F
    F --> G --> H --> I"""
    html = f"""<div style="display:flex;justify-content:center;">
      <div class="mermaid">{mermaid_code}</div></div>
      <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
      <script>mermaid.initialize({{startOnLoad:true, theme:"default"}});</script>"""
    st.components.v1.html(html, height=620, scrolling=True)

def workflow_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Architecture & Workflow</h2>", unsafe_allow_html=True)
    render_workflow_diagram()

def about_page():
    st.markdown('<h1 style="text-align:center;font-weight:800;"><span style="color:#1e73ff;">SmartCart</span><span style="color:#e53935;">-Web</span></h1>',unsafe_allow_html=True)
    st.markdown(APP_BRAND_LINE); st.markdown("---")
    st.subheader("Team JaiMataDi")
    cols = st.columns(3)
    for member, col in zip(TEAM, cols):
        with col:
            if os.path.exists(member["photo"]): st.image(member["photo"], width=220)
            st.markdown(f"**{member['name']}**")
            st.markdown(f"[LinkedIn]({member['linkedin']})")

# ---------- ROUTER ----------
if page.startswith("🏁"): start_page()
elif page.startswith("🧱"): build_model_page()
elif page.startswith("🛒"): menu_reco_page()
elif page.startswith("📦"): batch_page()
elif page.startswith("📊"): metrics_page()
elif page.startswith("🧩"): workflow_page()
else: about_page()
