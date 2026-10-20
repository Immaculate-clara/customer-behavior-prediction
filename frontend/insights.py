"""Plain-Python business rules and labels used by the dashboard (no Streamlit code)."""
from __future__ import annotations

ROC_AUC_TARGET = 0.80
RECALL_TARGET = 0.70
CLV_R2_TARGET = 0.60
CURRENCY = "£"

FEATURE_LABELS = {
    "recency_days": "Days since last order",
    "frequency": "Number of orders",
    "monetary_total": "Total spend",
    "monetary_avg": "Average order value",
    "avg_basket_size": "Average basket size",
    "avg_basket_lines": "Average lines per order",
    "tenure_days": "Customer age (days)",
    "interpurchase_mean_days": "Average days between orders",
    "interpurchase_std_days": "Variation in days between orders",
    "purchase_frequency_trend": "Ordering trend (recent vs earlier)",
    "seasonal_order_share": "Share of orders in peak season",
    "n_returns": "Number of returns",
    "return_rate": "Return rate",
    "historical_clv": "Net historical value",
    "discount_line_share": "Share of discounted items",
    "avg_price_index": "Average price paid vs typical",
    "engagement_score": "Engagement score",
    "category_entropy": "Variety of categories bought",
    "n_categories": "Number of categories bought",
}


def pretty_feature(name: str) -> str:
    """Readable label for a model feature name."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("affinity_"):
        return "Spend share: " + name.removeprefix("affinity_").replace("_", " ").title()
    return name.replace("_", " ").capitalize()


def money(value: float) -> str:
    """Format a currency amount with thousands separators, e.g. £12,345."""
    return f"{CURRENCY}{value:,.0f}"


def recommend(risk_level: str, segment_name: str) -> str:
    """Suggested retention action from churn risk and segment (simple, explainable rules)."""
    valuable = segment_name in {"Champions", "Loyal Customers", "Discount Bulk Buyers"}
    if risk_level == "high" and valuable:
        return "Top retention priority: personal outreach from an account manager and a tailored loyalty offer."
    if risk_level == "high":
        return "Send a targeted win-back offer on products this customer has bought before."
    if risk_level == "medium":
        return "Keep the customer warm: a reminder email with relevant products or a seasonal preview."
    if valuable:
        return "Healthy and valuable: reward loyalty and look for cross-sell opportunities."
    return "No retention action needed right now."