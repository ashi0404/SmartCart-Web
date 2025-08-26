import os
import streamlit as st
import pandas as pd
from collections import Counter
from typing import Optional

from data_loader import (
    load_csvs, build_items_and_tags, build_normalized_comatrix,
    save_artifact, load_artifact
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

# ====== About page constants (you can edit later) ======
APP_BRAND_LINE = (
    "We are **Team JaiMataDi**.\n\n"
    "We built **SmartCart** — a lightweight recommendation engine that turns static upsells "
    "into personalized suggestions across apps, web, and kiosks.\n\n"
    "In our tests, SmartCart improved **Recall@3** and **Precision@3** by **150%** over the baseline. "
    "That means customers see more relevant add-ons — and **Wings R Us** captures more revenue per order."
)

TEAM = [
    {
        "name": "Bhaskar Ranjan Karn",
        "linkedin": "https://www.linkedin.com/in/bhaskar-ranjan-karn/",
        "photo": os.path.join(BASE_DIR, "assets", "team", "bhaskar.jpg"),
    },
    {
        "name": "Astitva",
        "linkedin": "https://www.linkedin.com/in/astitva-07a338229/",
        "photo": os.path.join(BASE_DIR, "assets", "team", "astitva.jpg"),
    },
    {
        "name": "Sanjay Kumar",
        "linkedin": "https://www.linkedin.com/in/sanjay-kumar-39b73a239/",
        "photo": os.path.join(BASE_DIR, "assets", "team", "sanjay.jpg"),
    },
]

st.set_page_config(
    page_title="SmartCart-Web — Menu Recommender",
    page_icon="🍗",
    layout="wide"
)

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
    [
        "🏁 Start",
        "🧱 Build Model (first run)",
        "🛒 Menu & Recommendations",
        "📦 Batch Predict (CSV)",
        "📊 Metrics & Explore",
        "🧩 Architecture & Workflow",
        "ℹ️ About",
    ],
    label_visibility="collapsed"
)


# ---------- HELPERS ----------
@st.cache_data(show_spinner=False)
def load_all_csvs():
    return load_csvs()


@st.cache_data(show_spinner=True)
def prepare_artifacts(sample_n: Optional[int]):
    dfs = load_all_csvs()
    order = dfs["order"].copy()

    # Parse & clean items
    from data_loader import extract_item_names, clean_item_list
    order["ITEM_LIST"] = order["ORDERS"].apply(extract_item_names).apply(clean_item_list)

    item_type, item_feat, top_by_type, all_items = build_items_and_tags(order)
    co_norm = build_normalized_comatrix(order, sample_n=sample_n)

    known_lower = {itm.lower(): itm for itm in all_items}
    known_items_lower = list(known_lower.keys())

    art = {
        "item_type": item_type,
        "item_feat": item_feat,
        "top_by_type": top_by_type,
        "co_norm": co_norm,
        "known_items_lower": known_items_lower,
        "lower_to_orig": known_lower
    }
    save_artifact("artifacts.pkl", art)
    return art


def load_or_build_artifacts():
    art = load_artifact("artifacts.pkl")
    if art is None:
        st.warning("Artifacts not found. Go to **Build Model (first run)**.")
        return None
    return art


def app_brand_title():
    st.markdown(
        """
        <h1 class="brand-center">
          <span class="brand-blue">SmartCart</span><span class="brand-red">-Web</span>
        </h1>
        """,
        unsafe_allow_html=True
    )


# ---------- PAGES ----------
def start_page():
    app_brand_title()
    st.caption("")

    st.markdown("<h2 class='page-h2'>Welcome</h2>", unsafe_allow_html=True)
    st.write(
        "Select up to **3 menu items** and get **top-3 recommendations**. Background stays white, text is rich dark for clarity.")

    try:
        dfs = load_all_csvs()
    except Exception as e:
        st.error(f"CSV load error: {e}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders", f"{len(dfs['order']):,}")
    c2.metric("Customers", f"{len(dfs['customer']):,}")
    c3.metric("Items", f"138")
    c4.metric("Test Rows", f"{len(dfs['test']):,}")

    st.markdown("<hr class='rule'/>", unsafe_allow_html=True)
    header("Quick steps", "🧭")
    st.markdown("1) **Build Model** → 2) **Menu & Recommendations** → 3) (Optional) **Batch Predict**")


def build_model_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Build Model (first run)</h2>", unsafe_allow_html=True)
    st.write("Computes normalized co-occurrence + item tagging. Cached to disk for reuse.")

    sample = st.slider(
        "Sample N orders (speed vs quality)",
        100_000, 1_414_410, 250_000, step=50_000,
        help="Use ~200–300k for dev; full dataset if you have RAM/time."
    )
    if st.button("🚀 Build now"):
        with st.spinner("Crunching pairs & normalizing..."):
            _ = prepare_artifacts(sample_n=sample)
        st.success("Artifacts built and cached to `artifacts/artifacts.pkl`")


def menu_reco_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Menu & Recommendations</h2>", unsafe_allow_html=True)

    art = load_or_build_artifacts()
    if art is None:
        return

    item_type, item_feat = art["item_type"], art["item_feat"]
    top_by_type, co_norm = art["top_by_type"], art["co_norm"]
    known_items_lower, lower_to_orig = art["known_items_lower"], art["lower_to_orig"]
    all_items = list(lower_to_orig.values())

    st.caption("Menu (pick up to 3 items):")

    selected = st.multiselect(
        "Select items",
        options=all_items,
        default=[],
        help="Start typing to search. You can choose at most 3."
    )

    if len(selected) > 3:
        st.error("You selected more than 3 items. Only the first 3 will be used.")
        selected = selected[:3]

    topbar_badges(selected, limit=3)

    btn = st.button("🍽️ Recommend", use_container_width=False, type="secondary", disabled=(len(selected) == 0))

    if btn:
        cart = normalize_user_items(selected, known_items_lower, lower_to_orig)
        recs = enhanced_recommend(cart, co_norm, item_type, top_by_type, item_feat)

        if not recs:
            st.warning("No recommendations found. Try different items or rebuild the model.")
            return

        st.markdown("<h3 class='page-h3'>Top 3 Recommendations</h3>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        cols = [col_a, col_b, col_c]
        for idx, (it, score) in enumerate(recs, start=1):
            t = item_type.get(it, "other")
            with cols[idx - 1]:
                reco_card(idx, it, score, t)


def batch_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Batch Predict (CSV)</h2>", unsafe_allow_html=True)

    art = load_or_build_artifacts()
    if art is None:
        return

    st.caption("Reads `data/test_data_question.csv` and writes output under `artifacts/`")
    if st.button("Run batch on test_data_question.csv"):
        try:
            test_path = os.path.join(DATA_DIR, "test_data_question.csv")
            test_df = pd.read_csv(test_path)
        except Exception as e:
            st.error(f"Cannot read test_data_question.csv: {e}")
            return

        out = batch_predict(
            test_df,
            art["co_norm"], art["item_type"], art["top_by_type"], art["item_feat"],
            art["known_items_lower"], art["lower_to_orig"]
        )
        out_path = os.path.join(ART_DIR, "SmartCart_Recommendation_Output.csv")
        out.to_csv(out_path, index=False)
        st.success(f"Saved: {out_path}")
        st.dataframe(out.head(20))
        with open(out_path, "rb") as f:
            st.download_button(
                "Download CSV", f,
                file_name="SmartCart_Recommendation_Output.csv",
                mime="text/csv"
            )


def metrics_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Metrics & Explore</h2>", unsafe_allow_html=True)

    art = load_or_build_artifacts()
    if art is None:
        return

    item_type = art["item_type"]
    counts = Counter(item_type.values())
    st.bar_chart(pd.DataFrame.from_dict(counts, orient="index", columns=["Count"]))

    st.markdown("<h4 class='page-h4'>Top Items by Type</h4>", unsafe_allow_html=True)
    cols = st.columns(4)
    for t, col in zip(["main", "side", "dip", "drink"], cols):
        with col:
            st.markdown(f"**{t.title()}** {TYPE_EMOJI.get(t, '🍽️')}")
            for it, cnt in art["top_by_type"].get(t,[])[:130]:
                st.write(f"{icon_for_item(it)} {it} — {cnt}")


def render_workflow_diagram():
    """Mermaid v11-safe diagram: no emojis/HTML inside labels."""
    mermaid_code = """
flowchart TD
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
        I["Business Output (Excel Sheet with 3 Recos/Test Cart)"]
    end

    A --> B --> C --> D
    D --> E1 --> F
    D --> E2 --> F
    F --> G --> H --> I
    """
    html = f"""
    <div style="display:flex;justify-content:center;">
      <div class="mermaid" style="max-width:100%;overflow-x:auto;">
        {mermaid_code}
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme:"default"}});</script>
    """
    # Use components to avoid Streamlit sanitizing the script tag
    st.components.v1.html(html, height=620, scrolling=True)


def workflow_page():
    app_brand_title()
    st.markdown("<h2 class='page-h2'>Architecture & Workflow</h2>", unsafe_allow_html=True)
    st.caption(
        "High-level pipeline: preprocessing → co-occurrence features → scoring/filters → top-3 recommendations + evaluation.")
    render_workflow_diagram()


def about_page():
    st.markdown(
        '<h1 style="text-align:center; font-weight:800; margin-bottom:0.25rem;">'
        '<span style="color:#1e73ff;">SmartCart</span>'
        '<span style="color:#e53935;">-Web</span>'
        '</h1>',
        unsafe_allow_html=True
    )
    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    st.markdown(APP_BRAND_LINE)
    st.markdown("---")

    st.subheader("Team JaiMataDi")
    cols = st.columns(3)
    for member, col in zip(TEAM, cols):
        with col:
            if os.path.exists(member["photo"]):
                st.image(member["photo"], width=220)
            else:
                st.markdown(
                    '<div style="width:220px; height:220px; '
                    'background:#f2f2f2; border:1px solid #e5e5e5; '
                    'border-radius:12px; display:flex; align-items:center; '
                    'justify-content:center;">'
                    '<span style="color:#888;">Add photo: assets/team/{}.jpg</span>'
                    '</div>'.format(member["name"].split()[0].lower()),
                    unsafe_allow_html=True
                )
            st.markdown(f"**{member['name']}**")
            if member.get("role"):
                st.caption(member["role"])
            st.markdown(f"[LinkedIn]({member['linkedin']})")


# ---------- ROUTER ----------
if page.startswith("🏁"):
    start_page()
elif page.startswith("🧱"):
    build_model_page()
elif page.startswith("🛒"):
    menu_reco_page()
elif page.startswith("📦"):
    batch_page()
elif page.startswith("📊"):
    metrics_page()
elif page.startswith("🧩"):
    workflow_page()
else:
    about_page()