"""Plotly figure builders for the dashboard (pure functions: data in, figure out)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from insights import money, pretty_feature

RISK_COLORS = {"low": "#2a9d8f", "medium": "#e9a23b", "high": "#d62839"}
SEGMENT_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a", "#457b9d", "#8d99ae"]
UP_COLOR, DOWN_COLOR = "#d62839", "#2a9d8f"
NEUTRAL = "#457b9d"


def _style(fig: go.Figure, title: str | None = None, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, font=dict(size=16)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=55 if title else 15, b=10),
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def risk_distribution_chart(distribution: dict[str, int]) -> go.Figure:
    levels = ["low", "medium", "high"]
    counts = [int(distribution.get(level, 0)) for level in levels]
    fig = go.Figure(
        go.Bar(
            x=[level.title() for level in levels],
            y=counts,
            marker_color=[RISK_COLORS[level] for level in levels],
            text=[f"{count:,}" for count in counts],
            textposition="outside",
            hovertemplate="%{x} risk: %{y:,} customers<extra></extra>",
        )
    )
    fig.update_yaxes(title="Customers", rangemode="tozero")
    fig.update_xaxes(title="Churn risk level")
    return _style(fig, "Customers by churn risk")


def churn_histogram_chart(bins: list[dict[str, Any]], low: float = 0.33, high: float = 0.66) -> go.Figure:
    centers = [(item["start"] + item["end"]) / 2 for item in bins]
    width = bins[0]["end"] - bins[0]["start"] if bins else 0.05
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=[item["count"] for item in bins],
            width=width * 0.92,
            marker_color=NEUTRAL,
            hovertemplate="Churn probability %{x:.0%}: %{y:,} customers<extra></extra>",
        )
    )
    for position, label in ((low, "medium risk"), (high, "high risk")):
        fig.add_vline(x=position, line_dash="dash", line_color="#6c757d", annotation_text=label, annotation_position="top")
    fig.update_xaxes(title="Predicted churn probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Customers")
    return _style(fig, "How churn risk is spread across customers")


def segment_churn_chart(segments: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(segments).sort_values("avg_churn_probability")
    fig = go.Figure(
        go.Bar(
            y=frame["segment"],
            x=frame["avg_churn_probability"],
            orientation="h",
            marker_color=NEUTRAL,
            text=[f"{value:.0%}" for value in frame["avg_churn_probability"]],
            textposition="outside",
            hovertemplate="%{y}: average churn risk %{x:.0%}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Average predicted churn probability", tickformat=".0%", range=[0, 1.05])
    return _style(fig, "Average churn risk by segment")


def segment_revenue_chart(segments: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(segments).sort_values("expected_revenue")
    fig = go.Figure(
        go.Bar(
            y=frame["segment"],
            x=frame["expected_revenue"],
            orientation="h",
            marker_color="#2a9d8f",
            text=[money(value) for value in frame["expected_revenue"]],
            textposition="outside",
            hovertemplate="%{y}: expected revenue %{text}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Expected revenue, next 90 days", range=[0, float(frame["expected_revenue"].max()) * 1.2])
    return _style(fig, "Expected revenue by segment")


def segment_size_chart(segments: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(segments)
    fig = go.Figure(
        go.Pie(
            labels=frame["segment"],
            values=frame["customers"],
            hole=0.55,
            marker=dict(colors=SEGMENT_COLORS[: len(frame)]),
            textinfo="percent",
            hovertemplate="%{label}: %{value:,} customers<extra></extra>",
        )
    )
    return _style(fig, "Share of customers by segment")


def segment_heatmap(segments: list[dict[str, Any]], columns: list[str]) -> go.Figure:
    """Segment behaviour at a glance: colour = how high the average is compared with other segments."""
    rows = {item["segment"]: item["averages"] for item in segments}
    frame = pd.DataFrame(rows).T
    columns = [name for name in columns if name in frame.columns]
    values = frame[columns]
    spread = values.std(ddof=0).replace(0, 1)
    z_scores = (values - values.mean()) / spread
    fig = go.Figure(
        go.Heatmap(
            z=z_scores.to_numpy(),
            x=[pretty_feature(name) for name in columns],
            y=list(values.index),
            text=[[f"{value:,.2f}" for value in row] for row in values.to_numpy()],
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmid=0,
            showscale=False,
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Average behaviour of each segment (red = higher than other segments, blue = lower)", height=430)


def probability_gauge(probability: float, low: float = 0.33, high: float = 0.66) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(suffix="%", font=dict(size=42)),
            gauge=dict(
                axis=dict(range=[0, 100], ticksuffix="%"),
                bar=dict(color="#1d3557", thickness=0.28),
                steps=[
                    dict(range=[0, low * 100], color="#bfe5df"),
                    dict(range=[low * 100, high * 100], color="#f8e2b5"),
                    dict(range=[high * 100, 100], color="#f2b8bd"),
                ],
            ),
        )
    )
    return _style(fig, "Churn probability", height=270)


def shap_chart(top_features: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bars: red pushes churn risk up, green pushes it down."""
    items = sorted(top_features, key=lambda item: abs(item["shap_value"]))
    labels = [f"{pretty_feature(item['feature'])} = {item['value']:,.2f}" for item in items]
    values = [item["shap_value"] for item in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[UP_COLOR if value > 0 else DOWN_COLOR for value in values],
            hovertemplate="%{y}<br>effect on churn score: %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#495057")
    fig.update_xaxes(title="Effect on churn score (SHAP value)")
    return _style(fig, "Why this score? Red raises churn risk, green lowers it", height=max(300, 55 * len(items) + 120))


def metric_comparison_chart(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    target: float | None = None,
    group_column: str = "family",
) -> go.Figure:
    """Horizontal bar chart of one metric per model, coloured by model family."""
    frame = frame.dropna(subset=[metric]).sort_values(metric)
    fig = go.Figure()
    for index, (group, part) in enumerate(frame.groupby(group_column, sort=False)):
        fig.add_trace(
            go.Bar(
                y=part["model"],
                x=part[metric],
                orientation="h",
                name=str(group),
                marker_color=SEGMENT_COLORS[index % len(SEGMENT_COLORS)],
                text=[f"{value:.3f}" for value in part[metric]],
                textposition="outside",
            )
        )
    if target is not None:
        fig.add_vline(x=target, line_dash="dash", line_color="#d62839", annotation_text=f"target {target:g}", annotation_position="top")
    upper = max(float(frame[metric].max()), target or 0.0) if len(frame) else 1.0
    fig.update_xaxes(range=[min(0.0, float(frame[metric].min())) if len(frame) else 0.0, upper * 1.15])
    return _style(fig, title, height=max(280, 48 * len(frame) + 120))