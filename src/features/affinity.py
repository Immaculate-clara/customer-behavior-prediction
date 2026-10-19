"""Product category affinity features and the next-purchase-category target.

The Online Retail II dataset has no category column, so categories are derived
from product descriptions with an ordered keyword rule set (first match wins).
The mapping uses only product text, never customer behaviour, so it cannot leak
information about the target. Features still use history rows only.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.features.build_features import (
    _cutoff_from_config,
    _horizon_from_config,
    split_history,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CATEGORY = "Other"

# Ordered rules: the first matching category wins.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Seasonal & Christmas", (
        "CHRISTMAS", "XMAS", "SANTA", "ADVENT", "SNOWFLAKE", "SNOWMAN", "REINDEER",
        "EASTER", "HALLOWEEN", "STOCKING", "TINSEL", "BAUBLE", "WREATH",
    )),
    ("Candles & Lighting", (
        "CANDLE", "T-LIGHT", "TLIGHT", "LANTERN", "LIGHT", "LIGHTS", "LAMP", "INCENSE",
        "FAIRY LIGHTS", "NIGHT LIGHT", "CANDLEHOLDER",
    )),
    ("Jewellery & Accessories", (
        "NECKLACE", "BRACELET", "EARRING", "EARRINGS", "BROOCH", "RING", "JEWELLERY",
        "BEAD", "BEADS", "HAIR", "HAIRBAND", "HAIRCLIP", "SCARF", "PENDANT", "CHARM",
    )),
    ("Toys & Games", (
        "TOY", "DOLL", "GAME", "PUZZLE", "PLAY", "PLAYHOUSE", "SKITTLE",
        "DOMINOES", "SPACEBOY", "DOLLY", "TEDDY",
    )),
    ("Garden & Outdoor", (
        "GARDEN", "PLANT", "BIRDHOUSE", "WATERING", "SEED", "OUTDOOR", "PARASOL",
        "HERB",
    )),
    ("Kitchen & Dining", (
        "MUG", "CUP", "TEA", "TEAPOT", "TEACUP", "PLATE", "BOWL", "CAKE", "COOKIE", "JUG",
        "SPOON", "KITCHEN", "APRON", "BAKING", "CAKESTAND", "TRAY", "GLASS", "GLASSES",
        "EGG", "PICNIC", "CUTLERY", "COASTER", "NAPKIN", "JAM", "SPICE",
        "TEASPOON", "KNIFE", "FORK", "PLACEMAT", "CASSEROLE", "SAUCER", "DINNER",
    )),
    ("Stationery & Party", (
        "CARD", "CARDS", "WRAP", "NOTEBOOK", "PENCIL", "PEN", "STICKER", "STICKERS",
        "PAPER", "RIBBON", "GIFT", "TAG", "BUNTING", "PARTY", "BALLOON", "ALBUM",
        "STATIONERY", "ENVELOPE", "POSTCARD", "CALENDAR", "DIARY", "NAPKINS",
    )),
    ("Bags & Storage", (
        "BAG", "BASKET", "BOX", "BOXES", "TIN", "JAR", "CASE", "PURSE", "TRUNK", "CRATE",
        "CHEST", "DRAWER", "HOLDER", "ORGANISER", "RACK", "SHELF", "CABINET", "WALLET",
    )),
    ("Home Decor", (
        "FRAME", "CUSHION", "CLOCK", "MIRROR", "SIGN", "DOORMAT", "WALL", "HANGING",
        "VINTAGE", "HEART", "ORNAMENT", "DECORATION", "DOOR", "LETTER", "VASE",
        "RETROSPOT", "BLANKET", "THROW", "CURTAIN", "RUG", "PILLOW", "BATHROOM",
        "HOT WATER BOTTLE", "WARMER", "PLAQUE", "CHAIR", "STOOL",
    )),
]


def _compile(words: tuple[str, ...]) -> re.Pattern[str]:
    """Whole-word pattern that also tolerates a plural 'S'."""
    alternatives = "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})S?\b")


_COMPILED_RULES = [(name, _compile(words)) for name, words in _CATEGORY_KEYWORDS]
CATEGORIES: list[str] = [name for name, _ in _CATEGORY_KEYWORDS] + [DEFAULT_CATEGORY]


def category_slug(category: str) -> str:
    """Column-name friendly version, e.g. 'Kitchen & Dining' -> 'kitchen_dining'."""
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def categorize_descriptions(descriptions: pd.Series) -> pd.Series:
    """Assign a category to each description using the ordered keyword rules."""
    text = descriptions.fillna("").astype(str).str.upper()
    result = pd.Series(DEFAULT_CATEGORY, index=descriptions.index, dtype="object")
    unassigned = pd.Series(True, index=descriptions.index)
    for category, pattern in _COMPILED_RULES:
        hit = unassigned & text.str.contains(pattern, regex=True)
        result[hit] = category
        unassigned &= ~hit
    return result


def build_category_map(transactions: pd.DataFrame) -> pd.Series:
    """Map each stock_code to a category using its most frequent description."""
    described = transactions.dropna(subset=["description"])
    main_description = (
        described.groupby(["stock_code", "description"]).size().rename("n").reset_index()
        .sort_values(["stock_code", "n", "description"], ascending=[True, False, True])
        .drop_duplicates("stock_code")
        .set_index("stock_code")["description"]
    )
    return categorize_descriptions(main_description)


def build_affinity_features(
    transactions: pd.DataFrame,
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-customer share of spend in each category, using history before the cutoff."""
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date is not None else _cutoff_from_config()
    history = split_history(transactions, cutoff)
    sales = history[~history["is_cancellation"]]

    category_map = build_category_map(history)  # history only: no post-cutoff descriptions
    category = sales["stock_code"].map(category_map).fillna(DEFAULT_CATEGORY)

    spend = (
        sales.assign(category=category)
        .groupby(["customer_id", "category"])["line_total"].sum()
        .unstack(fill_value=0.0)
        .reindex(columns=CATEGORIES, fill_value=0.0)
        .clip(lower=0.0)
    )
    totals = spend.sum(axis=1).replace(0.0, np.nan)
    share = spend.div(totals, axis=0).fillna(0.0)

    probabilities = share.to_numpy()
    log_p = np.log(probabilities, out=np.zeros_like(probabilities), where=probabilities > 0)
    entropy = -(probabilities * log_p).sum(axis=1) / np.log(len(CATEGORIES))

    features = share.rename(columns={name: f"affinity_{category_slug(name)}" for name in CATEGORIES})
    features["top_category"] = share.idxmax(axis=1)
    features["n_categories"] = (share > 0).sum(axis=1)
    features["category_entropy"] = entropy
    logger.info("Built category affinity features for %s customers", f"{len(features):,}")
    return features


def build_next_category_labels(
    transactions: pd.DataFrame,
    customer_ids: pd.Index,
    cutoff_date: str | pd.Timestamp | None = None,
    horizon_days: int | None = None,
) -> pd.Series:
    """Top-spend category in the horizon window after the cutoff (NaN if no purchase)."""
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date is not None else _cutoff_from_config()
    horizon = int(horizon_days) if horizon_days is not None else _horizon_from_config()
    window_end = cutoff + pd.Timedelta(days=horizon)

    future = transactions[
        (transactions["invoice_date"] >= cutoff)
        & (transactions["invoice_date"] < window_end)
        & (~transactions["is_cancellation"])
    ]
    category_map = build_category_map(transactions)
    category = future["stock_code"].map(category_map).fillna(DEFAULT_CATEGORY)

    spend = (
        future.assign(category=category)
        .groupby(["customer_id", "category"])["line_total"].sum()
        .rename("spend").reset_index()
        .sort_values(["customer_id", "spend", "category"], ascending=[True, False, True])
        .drop_duplicates("customer_id")
        .set_index("customer_id")["category"]
    )
    return spend.reindex(customer_ids).rename("next_category")