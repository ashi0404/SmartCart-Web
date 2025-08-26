import streamlit as st

TYPE_EMOJI = {
    "main": "🍱",
    "side": "🍟",
    "dip": "🥣",
    "drink": "🥤",
    "other": "🍽️",
}

def icon_for_item(name: str) -> str:
    n = name.lower()
    if "wing" in n:
        return "🍗"
    if "fries" in n or "fry" in n:
        return "🍟"
    if "dip" in n or "sauce" in n or "ranch" in n:
        return "🥣"
    if "burger" in n or "sandwich" in n:
        return "🍔"
    if "corn" in n:
        return "🌽"
    if "drink" in n or "cola" in n or "juice" in n:
        return "🥤"
    return "🍽️"

def header(text: str, emoji: str = ""):
    st.markdown(f"<h4 class='section-title'>{emoji} {text}</h4>", unsafe_allow_html=True)

def topbar_badges(items, limit=3):
    shown = items[:limit]
    if not shown:
        return
    st.markdown("<div class='badge-bar'>", unsafe_allow_html=True)
    for it in shown:
        st.markdown(f"<span class='badge'>{icon_for_item(it)} {it}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def reco_card(rank: int, item_name: str, score: float, typ: str):
    emoji = icon_for_item(item_name)
    typ_emoji = TYPE_EMOJI.get(typ, "🍽️")
    html = f"""
    <div class="reco-card">
      <div class="reco-rank reco-rank-{rank}">{rank}</div>
      <div class="reco-name">{emoji} {item_name}</div>
      <div class="reco-meta">
        {'🥇' if rank==1 else '🥈' if rank==2 else '🥉'} Rec {rank} {typ_emoji} {typ.title()}
        &nbsp;•&nbsp; Confidence <strong>{score:.2f}</strong>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)