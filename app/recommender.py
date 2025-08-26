from __future__ import annotations
from collections import defaultdict, Counter
from difflib import get_close_matches
from typing import List, Dict, Tuple, Iterable, Optional

# Blacklist items we never recommend
DEFAULT_BLACKLIST = {
    "Plastic Fork","Plastic Knife","Plastic Straw","Plastic Utensils",
    "Delivery Fee","Unavailable Item","Ketchup Pack","Seasoning Pack","Extra Sauce"
}

def enhanced_recommend(cart_items: List[str],
                       co_dict: Dict[str, Dict[str, float]],
                       item_type: Dict[str,str],
                       top_items_by_type: Dict[str, List[Tuple[str,int]]],
                       item_tags: Dict[str,set],
                       blacklist: set[str] = DEFAULT_BLACKLIST,
                       top_n: int = 3,
                       boost_factor: float = 1.2,
                       max_per_type: int = 1) -> List[Tuple[str, float]]:
    """Type-aware, spicy-aware, fallback-enabled recommender returning (item, score)."""
    score = defaultdict(float)
    cart_types = Counter([item_type.get(x, "other") for x in cart_items])
    cart_has_spicy = any("spicy" in item_tags.get(x, set()) for x in cart_items)

    # 1) score from co-occurrence matrix
    for it in cart_items:
        if it not in co_dict:
            continue
        for co_it, cnt in co_dict[it].items():
            if co_it in cart_items or co_it in blacklist:
                continue
            t = item_type.get(co_it, "other")
            tags = item_tags.get(co_it, set())
            spicy_bonus = cnt*0.3 if ("spicy" in tags and not cart_has_spicy) else (cnt*0.1 if "spicy" in tags else 0.0)
            if cart_types.get(t, 0) == 0 and t == "drink":
                score[co_it] += cnt*(boost_factor*1.5) + spicy_bonus
            elif cart_types.get(t, 0) == 0:
                score[co_it] += cnt*boost_factor + spicy_bonus
            else:
                score[co_it] += cnt + spicy_bonus

    # 2) top-N with 1 per type
    sorted_items = sorted(score.items(), key=lambda x: x[1], reverse=True)
    reco, used_type = [], Counter()
    for it, sc in sorted_items:
        t = item_type.get(it, "other")
        if used_type[t] >= max_per_type:
            continue
        reco.append((it, round(float(sc), 4)))
        used_type[t] += 1
        if len(reco) >= top_n:
            break

    # 3) fallback fill
    if len(reco) < top_n:
        for t in ["main","side","dip","drink"]:
            if used_type[t] >= max_per_type:
                continue
            for cand, _ in top_items_by_type.get(t, []):
                if cand in cart_items or cand in [r[0] for r in reco] or cand in blacklist:
                    continue
                reco.append((cand, 0.0))
                used_type[t] += 1
                if len(reco) >= top_n: break
            if len(reco) >= top_n: break
    return reco[:top_n]

def normalize_user_items(raw_items: Iterable[str],
                         known_items_lower: List[str],
                         lower_to_orig: Dict[str,str],
                         cutoff: float = 0.75) -> List[str]:
    mapped = []
    for x in raw_items:
        if not isinstance(x, str) or not x.strip():
            continue
        lx = x.lower()
        if lx in lower_to_orig:
            mapped.append(lower_to_orig[lx]); continue
        m = get_close_matches(lx, known_items_lower, n=1, cutoff=cutoff)
        mapped.append(lower_to_orig[m[0]] if m else x)
    return mapped

def batch_predict(test_df,
                  co_dict, item_type, top_items_by_type, item_tags,
                  known_items_lower, lower_to_orig,
                  blacklist=DEFAULT_BLACKLIST, top_n=3):
    out = test_df.copy()
    for col in ["RECOMMENDATION 1","RECOMMENDATION 2","RECOMMENDATION 3"]:
        out[col] = ""
    for idx, row in out.iterrows():
        raw = [row.get("item1",""), row.get("item2",""), row.get("item3","")]
        cart = normalize_user_items(raw, known_items_lower, lower_to_orig)
        recs = enhanced_recommend(cart, co_dict, item_type, top_items_by_type, item_tags, blacklist, top_n=top_n)
        for i,(it,_) in enumerate(recs):
            out.at[idx, f"RECOMMENDATION {i+1}"] = it
    return out