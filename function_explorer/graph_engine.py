from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import plotly.graph_objects as go

from . import config


def generate_x_samples(x_min: float, x_max: float, count: int) -> List[float]:
    if count < 2:
        return [x_min]
    step = (x_max - x_min) / (count - 1)
    return [x_min + i * step for i in range(count)]


def generate_x_grid() -> List[float]:
    return generate_x_samples(config.X_MIN, config.X_MAX, config.NUM_SAMPLES)


def evaluate_quadratic(a: float, b: float, c: float, xs: Sequence[float]) -> List[float]:
    return [a * (x ** 2) + b * x + c for x in xs]


def discriminant(a: float, b: float, c: float) -> float:
    return b * b - 4 * a * c


def vertex(a: float, b: float, c: float, *, eps: float = config.EPS_ZERO) -> Tuple[Optional[float], Optional[float]]:
    if abs(a) < eps:
        return None, None
    xv = -b / (2 * a)
    yv = a * xv * xv + b * xv + c
    if not (math.isfinite(xv) and math.isfinite(yv)):
        return None, None
    return xv, yv


def real_roots(a: float, b: float, c: float, *, eps: float = config.EPS_ZERO) -> List[float]:
    if abs(a) < eps:
        if abs(b) < eps:
            return []
        root = -c / b
        return [root] if math.isfinite(root) else []
    disc = discriminant(a, b, c)
    if disc < -eps:
        return []
    if abs(disc) <= eps:
        root = -b / (2 * a)
        return [root] if math.isfinite(root) else []
    sqrt_disc = math.sqrt(disc)
    roots = [(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)]
    return [r for r in roots if math.isfinite(r)]


def base_traces(xs: Iterable[float], ys: Iterable[float]) -> List[go.Scatter]:
    return [
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="y = ax^2 + bx + c",
            line=dict(config.CURVE_LINE_STYLE),
        ),
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="Vertex",
            marker=dict(config.VERTEX_MARKER_STYLE),
            hovertemplate="Vertex<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="Zeros",
            marker=dict(config.ZERO_MARKER_STYLE),
            hovertemplate="Zero<br>x=%{x:.2f}<extra></extra>",
            showlegend=False,
        ),
    ]


def overlay_traces(capacity: int) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    for _ in range(capacity):
        traces.append(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Trace",
                line=dict(config.TRACE_LINE_STYLE),
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def build_figure(xs: Sequence[float], ys: Sequence[float], *, uirevision: str, trace_capacity: int) -> go.Figure:
    fig = go.Figure(
        data=[
            *base_traces(xs, ys),
            *overlay_traces(trace_capacity),
        ]
    )
    fig.update_layout(
        height=560,
        margin=dict(l=36, r=16, t=32, b=32),
        xaxis=dict(
            title="x",
            showgrid=True,
            zeroline=True,
            zerolinecolor=config.AXIS_LINE_STYLE["zerolinecolor"],
        ),
        yaxis=dict(
            title="y",
            showgrid=True,
            zeroline=True,
            zerolinecolor=config.AXIS_LINE_STYLE["zerolinecolor"],
        ),
        showlegend=False,
        uirevision=uirevision,
        shapes=[],
    )
    return fig
