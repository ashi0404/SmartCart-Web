from __future__ import annotations
import json, os, pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

def ensure_dirs():
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def _load_large_csv(path: str, chunksize: int = 200000) -> pd.DataFrame:
    """
    Load CSV in chunks to handle very large files (>700MB).
    Returns concatenated DataFrame.
    """
    dfs = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        dfs.append(chunk)
    return pd.concat(dfs, ignore_index=True)

def load_csvs() -> dict:
    paths = {
        "order": os.path.join(DATA_DIR, "order_data.csv"),
        "customer": os.path.join(DATA_DIR, "customer_data.csv"),
        "store": os.path.join(DATA_DIR, "store_data.csv"),
        "test": os.path.join(DATA_DIR, "test_data_question.csv"),
    }
    dfs = {}
    for key, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required CSV: {p}")
        # ✅ Use chunked reading for huge files
        dfs[key] = _load_large_csv(p)
    return dfs

NON_ITEMS = ["memo", "blankline", "asap", "order"]

def extract_item_names(order_str: str) -> list[str]:
    try:
        order_json = json.loads(order_str)
        item_names = []
        for order in order_json.get("orders", []):
            for item in order.get("item_details", []):
                nm = item.get("item_name")
                if nm:
                    item_names.append(nm)
        return item_names
    except Exception:
        return []

def clean_item_list(item_list: list[str]) -> list[str]:
    return [it for it in item_list if not any(sw in it.lower() for sw in NON_ITEMS)]

def tag_item_type(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["combo","feast","meal","wings","strips","flavor platter","sub","box","lunch","crispy"]):
        return "main"
    if "dip" in n or "sauce" in n:
        return "dip"
    if any(k in n for k in ["fries","corn","sticks","cake"]):
        return "side"
    if any(k in n for k in ["soda","tea","lemonade","drink","lager","punch","root beer","water"]):
        return "drink"
    return "other"

def extract_item_features(item_name: str) -> set[str]:
    n = item_name.lower()
    tags = set()
    if any(kw in n for kw in ["plastic","fork","knife","spoon","napkin","packaging","fee","delivery"]):
        tags.add("non-food"); return tags
    if any(veg_kw in n for veg_kw in ["veggie","veg","corn","celery","sticks","salad","carrot"]):
        tags.add("veg")
    else:
        tags.add("non-veg")
    if "spicy" in n:
        tags.add("spicy")
    if any(k in n for k in ["combo","feast","bundle","lunch","box","platter"]):
        tags.add("combo")
    if any(k in n for k in ["cake","dessert"]):
        tags.add("dessert")
    if any(k in n for k in ["soda","fruit punch","root beer","drink","lemonade","tea"]):
        tags.add("cold_drink")
    return tags

def build_items_and_tags(order_df: pd.DataFrame) -> tuple[dict, dict, dict, list[str]]:
    order_df = order_df.copy()
    order_df["ITEM_LIST"] = order_df["ORDERS"].apply(extract_item_names).apply(clean_item_list)

    all_items = sorted({it for row in order_df["ITEM_LIST"] for it in row})
    item_type_dict = {it: tag_item_type(it) for it in all_items}
    item_feature_dict = {it: extract_item_features(it) for it in all_items}

    cnt = Counter([it for row in order_df["ITEM_LIST"] for it in row])
    top_items_by_type = defaultdict(list)
    for item, c in cnt.items():
        t = item_type_dict.get(item, "other")
        if t in ["main","side","dip","drink"]:
            top_items_by_type[t].append((item, c))
    for t in top_items_by_type:
        top_items_by_type[t].sort(key=lambda x: -x[1])

    return item_type_dict, item_feature_dict, top_items_by_type, all_items

def build_normalized_comatrix(order_df: pd.DataFrame, sample_n: Optional[int] = None) -> dict:
    if sample_n:
        order_df = order_df.sample(n=min(sample_n, len(order_df)), random_state=42)

    lists = order_df["ITEM_LIST"]
    from collections import defaultdict
    item_count = defaultdict(int)
    pair_count = defaultdict(lambda: defaultdict(int))

    for items in lists:
        uniq = list(set(items))
        for i in uniq:
            item_count[i] += 1
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                if i == j:
                    continue
                a, b = uniq[i], uniq[j]
                pair_count[a][b] += 1

    norm = defaultdict(dict)
    for a in pair_count:
        denom = item_count[a] or 1
        for b, c in pair_count[a].items():
            norm[a][b] = c / denom
    return norm

def save_artifact(name: str, obj):
    ensure_dirs()
    p = os.path.join(ART_DIR, name)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return p

def load_artifact(name: str):
    p = os.path.join(ART_DIR, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None
