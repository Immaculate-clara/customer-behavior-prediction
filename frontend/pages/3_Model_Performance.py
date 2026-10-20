"""Model results: churn models, customer value (CLV) models and explainability figures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import loaders
import ui
from insights import CLV_R2_TARGET, RECALL_TARGET, ROC_AUC_TARGET

ui.setup_page("Model Performance", "🧪")

FIGURES_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"
comparison = loaders.model_comparison()

tab_churn, tab_clv, tab_explain = st.tabs(["Churn models", "Customer value (CLV)", "Explainability"])

# ---- Churn models ---------------------------------------------------------------------------
with tab_churn:
    classical = pd.DataFrame(comparison["churn"]).assign(family="Classical ML")
    deep = pd.DataFrame(comparison["deep_learning"]).assign(family="Deep learning")
    churn = pd.concat([classical, deep], ignore_index=True)
    if churn.empty:
        st.info("No churn results found. Run: python -m src.models.train_classical")
    else:
        st.caption("All models are scored on the same held-out test customers.")
        left, right = st.columns(2)
        with left:
            ui.show_figure(charts.metric_comparison_chart(churn, "test_roc_auc", "ROC-AUC (higher is better)", ROC_AUC_TARGET))
        with right:
            ui.show_figure(charts.metric_comparison_chart(churn, "test_recall", "Recall: share of churners caught", RECALL_TARGET))

        columns = ["family", "model", "test_roc_auc", "test_recall", "test_precision", "test_f1", "test_accuracy"]
        shown = churn[[name for name in columns if name in churn.columns]].sort_values("test_roc_auc", ascending=False)
        ui.show_table(shown.rename(columns=lambda name: name.replace("_", " ").capitalize()).round(3))
        best = shown.iloc[0]
        st.success(
            f"Best churn model by test ROC-AUC: **{best['model']}** ({best['test_roc_auc']:.3f}). "
            f"Targets: ROC-AUC >= {ROC_AUC_TARGET}, recall >= {RECALL_TARGET}."
        )

# ---- CLV models -----------------------------------------------------------------------------
with tab_clv:
    clv = pd.DataFrame(comparison["clv"])
    if clv.empty:
        st.info("No CLV results found. Run: python -m src.models.train_clv")
    else:
        clv = clv.assign(family="CLV regression")
        left, right = st.columns(2)
        with left:
            ui.show_figure(charts.metric_comparison_chart(clv, "test_r2_log", "R-squared on log revenue (higher is better)"))
        with right:
            ui.show_figure(charts.metric_comparison_chart(clv, "test_spearman", "Rank correlation (Spearman)"))
        columns = ["model", "val_r2_log", "test_r2_log", "test_r2", "test_mae", "test_rmse", "test_spearman"]
        shown = clv[[name for name in columns if name in clv.columns]].sort_values("val_r2_log", ascending=False)
        ui.show_table(shown.rename(columns=lambda name: name.replace("_", " ").capitalize()).round(3))
        models_only = clv[clv["model"] != "baseline_mean"]
        best_raw = float(models_only["test_r2"].max())
        if best_raw >= CLV_R2_TARGET:
            st.success(f"Target met: best raw-revenue R-squared is {best_raw:.2f} (target {CLV_R2_TARGET}).")
        else:
            st.warning(
                f"The raw-revenue R-squared target of {CLV_R2_TARGET} is not met (best: {best_raw:.2f}). "
                "About 57% of customers spend nothing in the window and a few very large accounts dominate the total, "
                "so raw R-squared is unstable. Models are therefore ranked on validation log-scale R-squared, "
                "and the rank correlation shows they order customers by value well."
            )

# ---- Explainability -------------------------------------------------------------------------
with tab_explain:
    figures = [
        ("shap_importance.png", "Average impact of each feature on churn predictions"),
        ("shap_summary.png", "How feature values push individual predictions up or down"),
        ("partial_dependence.png", "Average churn probability as a feature changes"),
    ]
    found = [(FIGURES_DIR / name, caption) for name, caption in figures if (FIGURES_DIR / name).exists()]
    if not found:
        st.info("No figures found. Run: python -m src.explainability.explain")
    for path, caption in found:
        st.image(str(path), caption=caption)