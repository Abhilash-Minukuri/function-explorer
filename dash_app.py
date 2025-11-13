"""Dash proof-of-concept layout for Function Explorer (Step 1)."""

from __future__ import annotations

import csv
import io
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import dash
from dash import Input, Output, State, dcc, html
import plotly.graph_objects as go

# Quadratic sampling grid
_X_MIN = -10.0
_X_MAX = 10.0
_NUM_SAMPLES = 401

_DEFAULT_PARAMS: Dict[str, float] = {"a": 1.0, "b": 0.0, "c": 0.0}
_UI_BASE_TOKEN = "quadratic-"
_DEFAULT_UI_NONCE = "0"
_DEFAULT_CONSENT_STATE: Dict[str, Any] = {
    "granted": False,
    "declined": False,
    "timestamp_utc": None,
}
_PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    "a": {"min": -5.0, "max": 5.0, "step": 0.1},
    "b": {"min": -10.0, "max": 10.0, "step": 0.1},
    "c": {"min": -10.0, "max": 10.0, "step": 0.1},
}
_SOURCE_STORE_DEFAULT: Dict[str, Any] = {
    "param": None,
    "source": None,
    "value": None,
    "t": None,
}

_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / "function_explorer" / "data"
_LOG_RATE_LIMIT_SECONDS = 0.1  # ~10 Hz
_MONOTONIC_ZERO = time.monotonic()
_SCHEMA_VERSION = 1
_FUNCTION_TYPE = "quadratic"
_APP_MODE = "dash-hybrid"
_DEFAULT_INTERACTION_PHASE = "change"
_THROTTLE_STATE: Dict[str, Dict[str, Any]] = {}
_SESSION_PARAM_CACHE: Dict[str, Dict[str, float]] = {}
_DEFAULT_OVERLAY_STATE = {"vertex_axis": True, "zeros": True}
_OVERLAY_STATE: Dict[str, Dict[str, bool]] = {}
_CSV_BASE_COLUMNS: List[str] = [
    "schema_version",
    "session_id",
    "timestamp",
    "function_type",
    "event",
    "param_name",
    "old_value",
    "new_value",
    "source",
    "elapsed_time",
    "mode",
    "interaction_phase",
    "viewport_xrange_min",
    "viewport_xrange_max",
    "viewport_yrange_min",
    "viewport_yrange_max",
    "uirevision",
]

_MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

_MODAL_OVERLAY_BASE_STYLE: Dict[str, Any] = {
    "position": "fixed",
    "inset": "0",
    "backgroundColor": "rgba(0, 0, 0, 0.4)",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "zIndex": 1000,
    "padding": "16px",
}

_MODAL_PANEL_STYLE: Dict[str, Any] = {
    "backgroundColor": "#ffffff",
    "padding": "24px",
    "borderRadius": "12px",
    "maxWidth": "640px",
    "width": "100%",
    "boxShadow": "0 8px 20px rgba(0,0,0,0.15)",
}


def _generate_x_samples(x_min: float, x_max: float, count: int) -> List[float]:
    if count < 2:
        return [x_min]
    step = (x_max - x_min) / (count - 1)
    return [x_min + i * step for i in range(count)]


_X_GRID = _generate_x_samples(_X_MIN, _X_MAX, _NUM_SAMPLES)


def _evaluate_quadratic(a: float, b: float, c: float, xs: List[float]) -> List[float]:
    return [a * (x ** 2) + b * x + c for x in xs]


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_session_id(session_data: Optional[Dict[str, Any]]) -> str:
    if isinstance(session_data, dict):
        raw = session_data.get("session_id")
        if isinstance(raw, str) and raw:
            return raw
    return "unknown"


def _default_consent_data() -> Dict[str, Any]:
    return dict(_DEFAULT_CONSENT_STATE)


def _is_logging_allowed(consent_data: Optional[Dict[str, Any]]) -> bool:
    return bool(consent_data and consent_data.get("granted"))


def _is_modal_dismissed(consent_data: Optional[Dict[str, Any]]) -> bool:
    if not consent_data:
        return False
    return bool(consent_data.get("granted") or consent_data.get("declined"))


def _modal_overlay_style(visible: bool) -> Dict[str, Any]:
    style = dict(_MODAL_OVERLAY_BASE_STYLE)
    style["display"] = "flex" if visible else "none"
    return style


def _extract_source_label(source_store: Optional[Dict[str, Any]]) -> str:
    if isinstance(source_store, dict):
        source = source_store.get("source")
        if source in {"slider", "input", "stepper"}:
            return str(source)
    return "slider"


def _safe_session_id(session_id: Optional[str]) -> str:
    return session_id if isinstance(session_id, str) and session_id else "unknown"


def _get_session_log_path(session_id: str) -> Path:
    safe_id = _safe_session_id(session_id)
    return _DATA_DIR / f"session_{safe_id}.jsonl"


def _write_log_record(session_id: str, record: Dict[str, Any]) -> None:
    safe_id = _safe_session_id(session_id)
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = _get_session_log_path(safe_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print("[dash-log]", exc, record)


def _flush_pending_record(session_id: str) -> None:
    state = _THROTTLE_STATE.get(session_id)
    if not state:
        return
    state["timer"] = None
    pending = state.get("pending")
    if not pending:
        return
    _write_log_record(session_id, pending)
    state["pending"] = None
    state["last_ts"] = time.monotonic()


def _log_with_throttle(session_id: str, record: Dict[str, Any]) -> None:
    state = _THROTTLE_STATE.setdefault(
        session_id,
        {"last_ts": 0.0, "pending": None, "timer": None},
    )
    now = time.monotonic()
    since_last = now - state["last_ts"]
    if since_last >= _LOG_RATE_LIMIT_SECONDS:
        _write_log_record(session_id, record)
        state["last_ts"] = now
        state["pending"] = None
        timer = state.get("timer")
        if timer:
            timer.cancel()
            state["timer"] = None
        return

    state["pending"] = record
    if state.get("timer"):
        return
    delay = max(_LOG_RATE_LIMIT_SECONDS - since_last, 0.01)
    timer = threading.Timer(delay, _flush_pending_record, args=(session_id,))
    timer.daemon = True
    state["timer"] = timer
    timer.start()


def _current_timestamp() -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return ts.replace("+00:00", "Z")


def _elapsed_ms() -> int:
    return int((time.monotonic() - _MONOTONIC_ZERO) * 1000)


def _resolve_uirevision_value(ui_store: Optional[Dict[str, Any]]) -> str:
    nonce = _DEFAULT_UI_NONCE
    if isinstance(ui_store, dict):
        raw = ui_store.get("uirevision_nonce")
        if raw is not None:
            nonce = str(raw)
    return f"{_UI_BASE_TOKEN}{nonce}"


def _base_log_record(
    session_id: str,
    *,
    event: str,
    param_name: Optional[str] = None,
    old_value: Optional[Any] = None,
    new_value: Optional[Any] = None,
    source: str = "system",
    uirevision: Optional[str] = None,
    viewport: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "session_id": _safe_session_id(session_id),
        "timestamp": _current_timestamp(),
        "function_type": _FUNCTION_TYPE,
        "event": event,
        "param_name": param_name,
        "old_value": old_value,
        "new_value": new_value,
        "source": source,
        "elapsed_time": _elapsed_ms(),
        "mode": _APP_MODE,
        "interaction_phase": _DEFAULT_INTERACTION_PHASE,
        "viewport": viewport,
        "uirevision": uirevision,
    }
    if extras:
        record.update(extras)
    return record


def _get_param_cache(session_id: str) -> Dict[str, float]:
    cache = _SESSION_PARAM_CACHE.setdefault(session_id, dict(_DEFAULT_PARAMS))
    return cache


def _get_overlay_state(session_id: str) -> Dict[str, bool]:
    state = _OVERLAY_STATE.setdefault(session_id, dict(_DEFAULT_OVERLAY_STATE))
    return state


def _read_session_log_records(session_id: str) -> List[Dict[str, Any]]:
    path = _get_session_log_path(session_id)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        print("[dash-log-read]", exc)
    return records


def _flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(record)
    viewport = flat.pop("viewport", None)
    x_min = x_max = y_min = y_max = None
    if isinstance(viewport, dict):
        xrange = viewport.get("xrange")
        yrange = viewport.get("yrange")
        if isinstance(xrange, Sequence) and len(xrange) == 2:
            x_min, x_max = xrange
        if isinstance(yrange, Sequence) and len(yrange) == 2:
            y_min, y_max = yrange
    flat["viewport_xrange_min"] = x_min
    flat["viewport_xrange_max"] = x_max
    flat["viewport_yrange_min"] = y_min
    flat["viewport_yrange_max"] = y_max
    return flat


def _build_csv_content(records: List[Dict[str, Any]]) -> Optional[str]:
    if not records:
        return None
    rows = [_flatten_record_for_csv(rec) for rec in records]
    columns = list(_CSV_BASE_COLUMNS)
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col) for col in columns})
    return buffer.getvalue()


def _is_toggle_enabled(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return "on" in value or True in value
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return bool(value)


def _format_value_preview(value: Optional[Any]) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def _append_preview_log(log_data: Any, message: str) -> List[str]:
    if isinstance(log_data, list):
        entries = log_data[-4:]
    else:
        entries = []
    entries = list(entries)
    entries.append(message)
    return entries[-5:]


def _format_preview_message(record: Dict[str, Any]) -> str:
    event = record.get("event", "event")
    if event == "param_change":
        param = record.get("param_name", "?")
        old_v = _format_value_preview(record.get("old_value"))
        new_v = _format_value_preview(record.get("new_value"))
        source = record.get("source", "source")
        return f"param_change: {param} {old_v} → {new_v} ({source})"
    if event == "overlay_toggle":
        overlay = record.get("overlay", "?")
        enabled = record.get("enabled")
        return f"overlay_toggle: {overlay} enabled={bool(enabled)}"
    if event == "reset":
        return "reset: parameters restored"
    if event == "consent":
        status = record.get("consent_status", "status")
        return f"consent: {status}"
    if event == "export":
        export_type = record.get("export_type", "unknown")
        return f"export: {export_type}"
    return event


def _build_consent_modal() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Research Consent"),
                    html.P(
                        "We're running a focused research study on Function Explorer. "
                        "With your consent, we log slider adjustments to understand how people explore quadratics."
                    ),
                    html.P(
                        "Each record includes the anonymous session ID stored in your browser, a timestamp, "
                        "and the current values of a, b, and c."
                    ),
                    html.P(
                        "You can download or erase your session data from the app menu later. "
                        "Declining keeps the app fully usable but disables logging."
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Accept",
                                id="btn-consent-accept",
                                n_clicks=0,
                                style={
                                    "marginRight": "12px",
                                    "padding": "8px 20px",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Button(
                                "Decline",
                                id="btn-consent-decline",
                                n_clicks=0,
                                style={"padding": "8px 20px"},
                            ),
                        ],
                        style={"marginTop": "18px", "textAlign": "right"},
                    ),
                ],
                style=_MODAL_PANEL_STYLE,
            )
        ],
        id="consent-modal",
        style=_modal_overlay_style(True),
    )


def _bump_uirevision_store(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(data) if isinstance(data, dict) else {}
    raw = base.get("uirevision_nonce", _DEFAULT_UI_NONCE)
    try:
        nonce_int = int(raw)
    except (TypeError, ValueError):
        try:
            nonce_int = int(float(raw))
        except (TypeError, ValueError):
            nonce_int = int(_DEFAULT_UI_NONCE)
    base["uirevision_nonce"] = str(nonce_int + 1)
    return base


def _build_figure(params: Dict[str, float], xs: List[float], uirevision: str) -> go.Figure:
    ys = _evaluate_quadratic(params["a"], params["b"], params["c"], xs)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="y = ax^2 + bx + c",
            ),
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                name="Vertex",
                marker=dict(color="#EF553B", size=10, line=dict(color="#ffffff", width=1)),
                hovertemplate="Vertex<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                name="Zeros",
                marker=dict(color="#00b5ad", size=9, symbol="x", line=dict(color="#ffffff", width=1)),
                hovertemplate="Zero<br>x=%{x:.2f}<extra></extra>",
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        height=560,
        margin=dict(l=36, r=16, t=32, b=32),
        xaxis=dict(
            title="x",
            showgrid=True,
            zeroline=True,
            zerolinecolor="#999999",
        ),
        yaxis=dict(
            title="y",
            showgrid=True,
            zeroline=True,
            zerolinecolor="#999999",
        ),
        showlegend=False,
        uirevision=uirevision,
        shapes=[],
    )
    return fig


_INITIAL_FIGURE = _build_figure(
    _DEFAULT_PARAMS,
    _X_GRID,
    f"{_UI_BASE_TOKEN}{_DEFAULT_UI_NONCE}",
)


app = dash.Dash(__name__, external_scripts=[_MATHJAX_CDN])
server = app.server


def _param_control_row(param: str, label: str, *, marks: Dict[float, str]) -> html.Div:
    cfg = _PARAM_BOUNDS[param]
    slider_id = f"slider-{param}"
    input_id = f"input-{param}"
    minus_id = f"btn-{param}-minus"
    plus_id = f"btn-{param}-plus"
    return html.Div(
        [
            html.Label(label, htmlFor=slider_id, style={"fontWeight": 600}),
            html.Div(
                [
                    html.Div(
                        dcc.Slider(
                            id=slider_id,
                            min=cfg["min"],
                            max=cfg["max"],
                            step=cfg["step"],
                            value=_DEFAULT_PARAMS[param],
                            marks=marks,
                            updatemode="drag",
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        style={"flex": "1"},
                    ),
                    dcc.Input(
                        id=input_id,
                        type="number",
                        min=cfg["min"],
                        max=cfg["max"],
                        step=cfg["step"],
                        value=_DEFAULT_PARAMS[param],
                        style={
                            "width": "90px",
                            "marginLeft": "12px",
                            "marginRight": "8px",
                        },
                    ),
                    html.Button(
                        "-",
                        id=minus_id,
                        n_clicks=0,
                        style={
                            "width": "44px",
                            "height": "44px",
                            "marginRight": "4px",
                        },
                    ),
                    html.Button(
                        "+",
                        id=plus_id,
                        n_clicks=0,
                        style={"width": "44px", "height": "44px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginTop": "8px",
                },
            ),
        ],
        style={"marginBottom": "24px"},
    )


def _serve_layout() -> html.Div:
    controls_column = html.Div(
        [
            html.H2("Controls"),
            _param_control_row("a", "a", marks={-5.0: "-5", 0.0: "0", 5.0: "5"}),
            _param_control_row(
                "b",
                "b",
                marks={-10.0: "-10", -5.0: "-5", 0.0: "0", 5.0: "5", 10.0: "10"},
            ),
            _param_control_row(
                "c",
                "c",
                marks={-10.0: "-10", -5.0: "-5", 0.0: "0", 5.0: "5", 10.0: "10"},
            ),
            html.Button(
                "Reset view & params",
                id="btn-reset",
                n_clicks=0,
                style={"marginTop": "8px"},
            ),
            html.Div(
                dcc.Checklist(
                    id="toggle-vertex",
                    options=[{"label": "Vertex & axis", "value": "on"}],
                    value=["on"],
                    labelStyle={"display": "flex", "alignItems": "center", "gap": "6px"},
                    inputStyle={"marginRight": "6px"},
                ),
                style={"marginTop": "16px"},
            ),
            html.Div(
                dcc.Checklist(
                    id="toggle-zeros",
                    options=[{"label": "Zeros (x-intercepts)", "value": "on"}],
                    value=["on"],
                    labelStyle={"display": "flex", "alignItems": "center", "gap": "6px"},
                    inputStyle={"marginRight": "6px"},
                ),
                style={"marginTop": "8px"},
            ),
        ],
        style={"flex": "1", "minWidth": "280px"},
    )

    graph_column = html.Div(
        [
            html.H2("Graph"),
            dcc.Graph(
                id="graph-main",
                figure=_INITIAL_FIGURE,
                config={"displaylogo": False},
            ),
            dcc.Markdown(
                "Recent logs will appear here.",
                id="log-display",
                style={"marginTop": "16px", "fontSize": "0.9rem"},
            ),
            html.Div(
                [
                    html.Button("Download JSONL", id="btn-download-jsonl", n_clicks=0),
                    html.Button(
                        "Download CSV",
                        id="btn-download-csv",
                        n_clicks=0,
                        style={"marginLeft": "8px"},
                    ),
                    dcc.Download(id="download-jsonl"),
                    dcc.Download(id="download-csv"),
                ],
                style={"marginTop": "16px"},
            ),
        ],
        style={"flex": "2", "minWidth": "0"},
    )

    main_content = html.Div(
        [controls_column, graph_column],
        style={
            "display": "flex",
            "gap": "32px",
            "alignItems": "flex-start",
        },
    )

    representations_section = html.Div(
        [
            html.H2("Representations"),
            html.Div(
                [
                    html.H3("Equation"),
                    dcc.Markdown(
                        "y = x²",
                        id="equation-view",
                        style={"fontSize": "1.15rem", "marginTop": "8px"},
                    ),
                    html.Div(
                        "",
                        id="verbal-a-tip",
                        style={
                            "fontSize": "0.9rem",
                            "color": "#444444",
                            "minHeight": "1.5em",
                            "marginTop": "4px",
                            "opacity": "0",
                            "visibility": "hidden",
                            "transition": "opacity 0.2s ease-in-out",
                        },
                    ),
                    html.Div(
                        "",
                        id="vertex-notice",
                        style={
                            "fontSize": "0.9rem",
                            "color": "#555555",
                            "minHeight": "1.2em",
                            "marginTop": "4px",
                        },
                    ),
                    html.Div(
                        "",
                        id="zeros-notice",
                        style={
                            "fontSize": "0.9rem",
                            "color": "#555555",
                            "minHeight": "1.2em",
                            "marginTop": "4px",
                        },
                    ),
                ],
                style={"marginTop": "16px"},
            ),
        ],
        style={"padding": "0 32px 32px"},
    )

    return html.Div(
        [
            dcc.Store(id="store-x", data=_X_GRID),
            dcc.Store(
                id="store-session",
                storage_type="local",
                data={"session_id": uuid.uuid4().hex},
            ),
            dcc.Store(id="store-ui", data={"uirevision_nonce": _DEFAULT_UI_NONCE}),
            dcc.Store(id="store-log-sink", data=[]),
            dcc.Store(
                id="store-consent",
                storage_type="local",
                data=_default_consent_data(),
            ),
            dcc.Store(id="store-source", data=_SOURCE_STORE_DEFAULT),
            dcc.Store(
                id="store-a-prev",
                data={"a_prev": _DEFAULT_PARAMS["a"], "visible_until": None, "last_rule": None},
            ),
            dcc.Interval(id="interval-verbal-a", interval=500, n_intervals=0, disabled=True),
            html.Div(main_content, style={"padding": "32px"}),
            representations_section,
            _build_consent_modal(),
        ]
    )


app.layout = _serve_layout


@app.callback(
    Output("store-consent", "data"),
    Input("btn-consent-accept", "n_clicks"),
    Input("btn-consent-decline", "n_clicks"),
    State("store-consent", "data"),
    State("store-session", "data"),
    State("store-ui", "data"),
    prevent_initial_call=True,
)
def _handle_consent_decision(accept_clicks, decline_clicks, consent_data, session_data, ui_store_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    data = dict(consent_data) if isinstance(consent_data, dict) else _default_consent_data()
    data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if trigger_id == "btn-consent-accept":
        data["granted"] = True
        data["declined"] = False
        status = "accepted"
    else:
        data["granted"] = False
        data["declined"] = True
        status = "declined"

    session_id = _get_session_id(session_data)
    record = _base_log_record(
        session_id,
        event="consent",
        param_name=None,
        old_value=None,
        new_value=None,
        source="button",
        uirevision=_resolve_uirevision_value(ui_store_data),
        viewport=None,
        extras={"consent_status": status},
    )
    _write_log_record(session_id, record)
    return data


@app.callback(
    Output("consent-modal", "style"),
    Input("store-consent", "data"),
)
def _toggle_consent_modal(consent_data):
    visible = not _is_modal_dismissed(consent_data)
    return _modal_overlay_style(visible)


app.clientside_callback(
    """
    (function() {
        const PARAM_CFG = {
            a: {min: -5.0, max: 5.0, step: 0.1},
            b: {min: -10.0, max: 10.0, step: 0.1},
            c: {min: -10.0, max: 10.0, step: 0.1},
        };
        const DEFAULTS = {a: 1.0, b: 0.0, c: 0.0};
        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
        const roundStep = (value, step) => {
            const scaled = Math.round(value / step) * step;
            return Number(scaled.toFixed(6));
        };
        const toNumber = (value, fallback) => {
            const num = Number(value);
            return isFinite(num) ? num : fallback;
        };

        return function(
            aMinus,
            aPlus,
            bMinus,
            bPlus,
            cMinus,
            cPlus,
            sliderA,
            sliderB,
            sliderC,
            inputA,
            inputB,
            inputC
        ) {
            const ctx =
                (window.dash_clientside && window.dash_clientside.callback_context) || {
                    triggered: [],
                };
            if (!ctx.triggered.length) {
                return [
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                ];
            }

            const trigger = ctx.triggered[0].prop_id.split(".")[0];
            const mapping = {
                "btn-a-minus": {param: "a", delta: -PARAM_CFG.a.step},
                "btn-a-plus": {param: "a", delta: PARAM_CFG.a.step},
                "btn-b-minus": {param: "b", delta: -PARAM_CFG.b.step},
                "btn-b-plus": {param: "b", delta: PARAM_CFG.b.step},
                "btn-c-minus": {param: "c", delta: -PARAM_CFG.c.step},
                "btn-c-plus": {param: "c", delta: PARAM_CFG.c.step},
            };

            const meta = mapping[trigger];
            if (!meta) {
                return [
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                ];
            }

            const sliderVals = {a: sliderA, b: sliderB, c: sliderC};
            const inputVals = {a: inputA, b: inputB, c: inputC};
            const current = toNumber(sliderVals[meta.param], toNumber(inputVals[meta.param], DEFAULTS[meta.param]));
            const cfg = PARAM_CFG[meta.param];
            const nextValue = roundStep(clamp(current + meta.delta, cfg.min, cfg.max), cfg.step);

            const sliderOutputs = [
                window.dash_clientside.no_update,
                window.dash_clientside.no_update,
                window.dash_clientside.no_update,
            ];
            const inputOutputs = [
                window.dash_clientside.no_update,
                window.dash_clientside.no_update,
                window.dash_clientside.no_update,
            ];
            const indexMap = {a: 0, b: 1, c: 2};
            const idx = indexMap[meta.param];
            sliderOutputs[idx] = nextValue;
            inputOutputs[idx] = nextValue;

            return [
                sliderOutputs[0],
                sliderOutputs[1],
                sliderOutputs[2],
                inputOutputs[0],
                inputOutputs[1],
                inputOutputs[2],
                {
                    param: meta.param,
                    source: "stepper",
                    value: nextValue,
                    t: Date.now(),
                },
            ];
        };
    })()
    """,
    [
        Output("slider-a", "value", allow_duplicate=True),
        Output("slider-b", "value", allow_duplicate=True),
        Output("slider-c", "value", allow_duplicate=True),
        Output("input-a", "value", allow_duplicate=True),
        Output("input-b", "value", allow_duplicate=True),
        Output("input-c", "value", allow_duplicate=True),
        Output("store-source", "data", allow_duplicate=True),
    ],
    [
        Input("btn-a-minus", "n_clicks"),
        Input("btn-a-plus", "n_clicks"),
        Input("btn-b-minus", "n_clicks"),
        Input("btn-b-plus", "n_clicks"),
        Input("btn-c-minus", "n_clicks"),
        Input("btn-c-plus", "n_clicks"),
    ],
    [
        State("slider-a", "value"),
        State("slider-b", "value"),
        State("slider-c", "value"),
        State("input-a", "value"),
        State("input-b", "value"),
        State("input-c", "value"),
    ],
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    (function() {
        const PARAM_CFG = {
            a: {min: -5.0, max: 5.0, step: 0.1},
            b: {min: -10.0, max: 10.0, step: 0.1},
            c: {min: -10.0, max: 10.0, step: 0.1},
        };
        const DEFAULTS = {a: 1.0, b: 0.0, c: 0.0};
        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
        const roundStep = (value, step) => {
            const scaled = Math.round(value / step) * step;
            return Number(scaled.toFixed(6));
        };

        const toNumber = (value, fallback) => {
            const num = Number(value);
            return isFinite(num) ? num : fallback;
        };

        return function(aInput, bInput, cInput, sliderA, sliderB, sliderC) {
            const sliderValues = {
                a: toNumber(sliderA, DEFAULTS.a),
                b: toNumber(sliderB, DEFAULTS.b),
                c: toNumber(sliderC, DEFAULTS.c),
            };

            const inputs = {a: aInput, b: bInput, c: cInput};
            const sliderOutputs = [
                window.dash_clientside.no_update,
                window.dash_clientside.no_update,
                window.dash_clientside.no_update,
            ];

            const paramOrder = ["a", "b", "c"];
            let changedParam = null;
            paramOrder.forEach((key, idx) => {
                const raw = Number(inputs[key]);
                if (!isFinite(raw)) {
                    return;
                }
                const sanitized = roundStep(clamp(raw, PARAM_CFG[key].min, PARAM_CFG[key].max), PARAM_CFG[key].step);
                if (!isFinite(sanitized)) {
                    return;
                }
                if (Math.abs(sanitized - sliderValues[key]) > 1e-9) {
                    sliderOutputs[idx] = sanitized;
                    sliderValues[key] = sanitized;
                    changedParam = key;
                }
            });

            let storeUpdate = window.dash_clientside.no_update;
            if (changedParam !== null) {
                storeUpdate = {
                    param: changedParam,
                    source: "input",
                    value: sliderValues[changedParam],
                    t: Date.now(),
                };
            }

            return [sliderOutputs[0], sliderOutputs[1], sliderOutputs[2], storeUpdate];
        };
    })()
    """,
    [
        Output("slider-a", "value", allow_duplicate=True),
        Output("slider-b", "value", allow_duplicate=True),
        Output("slider-c", "value", allow_duplicate=True),
        Output("store-source", "data", allow_duplicate=True),
    ],
    [
        Input("input-a", "value"),
        Input("input-b", "value"),
        Input("input-c", "value"),
    ],
    [
        State("slider-a", "value"),
        State("slider-b", "value"),
        State("slider-c", "value"),
    ],
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    (function() {
        const PARAM_CFG = {
            a: {min: -5.0, max: 5.0, step: 0.1},
            b: {min: -10.0, max: 10.0, step: 0.1},
            c: {min: -10.0, max: 10.0, step: 0.1},
        };
        const DEFAULTS = {a: 1.0, b: 0.0, c: 0.0};
        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
        const roundStep = (value, step) => {
            const scaled = Math.round(value / step) * step;
            return Number(scaled.toFixed(6));
        };

        if (!window.__paramSyncState) {
            window.__paramSyncState = {slider: {a: null, b: null, c: null}};
        }

        return function(aSlider, bSlider, cSlider, sourceStore) {
            const values = {
                a: roundStep(clamp(Number(aSlider ?? DEFAULTS.a), PARAM_CFG.a.min, PARAM_CFG.a.max), PARAM_CFG.a.step),
                b: roundStep(clamp(Number(bSlider ?? DEFAULTS.b), PARAM_CFG.b.min, PARAM_CFG.b.max), PARAM_CFG.b.step),
                c: roundStep(clamp(Number(cSlider ?? DEFAULTS.c), PARAM_CFG.c.min, PARAM_CFG.c.max), PARAM_CFG.c.step),
            };

            const outputs = [values.a, values.b, values.c];
            let changedParam = null;
            Object.keys(values).forEach((key) => {
                const prev = window.__paramSyncState.slider[key];
                if (prev === null || Math.abs(values[key] - prev) > 1e-9) {
                    changedParam = key;
                    window.__paramSyncState.slider[key] = values[key];
                }
            });

            let storeUpdate = window.dash_clientside.no_update;
            if (changedParam !== null) {
                const prevStore = sourceStore || {};
                const prevVal = Number(prevStore.value);
                const suppress =
                    prevStore &&
                    (prevStore.source === "input" || prevStore.source === "stepper") &&
                    prevStore.param === changedParam &&
                    isFinite(prevVal) &&
                    Math.abs(prevVal - values[changedParam]) < 1e-9;
                if (!suppress) {
                    storeUpdate = {
                        param: changedParam,
                        source: "slider",
                        value: values[changedParam],
                        t: Date.now(),
                    };
                }
            }

            return [outputs[0], outputs[1], outputs[2], storeUpdate];
        };
    })()
    """,
    [
        Output("input-a", "value", allow_duplicate=True),
        Output("input-b", "value", allow_duplicate=True),
        Output("input-c", "value", allow_duplicate=True),
        Output("store-source", "data", allow_duplicate=True),
    ],
    [
        Input("slider-a", "value"),
        Input("slider-b", "value"),
        Input("slider-c", "value"),
    ],
    [State("store-source", "data")],
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(
        aSlider,
        bSlider,
        cSlider,
        aInput,
        bInput,
        cInput,
        toggleVertexValue,
        toggleZerosValue,
        uiStore,
        figure,
        xs
    ) {
        const noUpdate = window.dash_clientside.no_update;
        if (!Array.isArray(xs) || xs.length === 0) {
            return [noUpdate, noUpdate, noUpdate];
        }

        const fallback = {a: 1.0, b: 0.0, c: 0.0};
        const resolveValue = (sliderVal, inputVal, key) => {
            let val = Number(sliderVal);
            if (!isFinite(val)) {
                val = Number(inputVal);
            }
            if (!isFinite(val)) {
                val = fallback[key];
            }
            return val;
        };

        var aNum = resolveValue(aSlider, aInput, "a");
        var bNum = resolveValue(bSlider, bInput, "b");
        var cNum = resolveValue(cSlider, cInput, "c");
        if (!isFinite(aNum) || !isFinite(bNum) || !isFinite(cNum)) {
            return [noUpdate, noUpdate, noUpdate];
        }

        if (!figure || !Array.isArray(figure.data) || figure.data.length === 0) {
            return [noUpdate, noUpdate, noUpdate];
        }

        var baseTrace = figure.data[0];
        if (!baseTrace) {
            return [noUpdate, noUpdate, noUpdate];
        }

        const isToggleOn = (value) => {
            if (Array.isArray(value)) {
                return value.indexOf("on") !== -1;
            }
            if (typeof value === "boolean") {
                return value;
            }
            if (value === null || value === undefined) {
                return false;
            }
            return Boolean(value);
        };

        const vertexToggleActive = isToggleOn(toggleVertexValue);
        const zerosToggleActive = isToggleOn(toggleZerosValue);

        var nonce = "0";
        if (uiStore && uiStore.uirevision_nonce !== undefined && uiStore.uirevision_nonce !== null) {
            nonce = String(uiStore.uirevision_nonce);
        }
        var uiRevision = "quadratic-" + nonce;

        var ys = new Array(xs.length);
        for (var i = 0; i < xs.length; i++) {
            var xVal = Number(xs[i]);
            if (!isFinite(xVal)) {
                ys[i] = null;
            } else {
                ys[i] = aNum * xVal * xVal + bNum * xVal + cNum;
            }
        }

        var newData = figure.data.slice();
        var newTrace = Object.assign({}, baseTrace);
        newTrace.y = ys;
        newData[0] = newTrace;

        var vertexTrace = newData[1];
        if (vertexTrace) {
            vertexTrace = Object.assign({}, vertexTrace);
        } else {
            vertexTrace = {
                x: [],
                y: [],
                mode: "markers",
                name: "Vertex",
                marker: {color: "#EF553B", size: 10, line: {color: "#ffffff", width: 1}},
                hovertemplate: "Vertex<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                showlegend: false,
            };
        }
        vertexTrace.mode = "markers";
        vertexTrace.showlegend = false;
        vertexTrace.hovertemplate = "Vertex<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>";

        var zerosTrace = newData[2];
        if (zerosTrace) {
            zerosTrace = Object.assign({}, zerosTrace);
        } else {
            zerosTrace = {
                x: [],
                y: [],
                mode: "markers",
                name: "Zeros",
                marker: {color: "#00b5ad", size: 9, symbol: "x", line: {color: "#ffffff", width: 1}},
                hovertemplate: "Zero<br>x=%{x:.2f}<extra></extra>",
                showlegend: false,
            };
        }
        zerosTrace.mode = "markers";
        zerosTrace.showlegend = false;
        if (!zerosTrace.marker) {
            zerosTrace.marker = {color: "#00b5ad", size: 9, symbol: "x", line: {color: "#ffffff", width: 1}};
        }
        zerosTrace.marker.symbol = zerosTrace.marker.symbol || "x";
        zerosTrace.hovertemplate = "Zero<br>x=%{x:.2f}<extra></extra>";

        const roundOneDecimal = (value) => {
            const rounded = Math.round(value * 10) / 10;
            return Object.is(rounded, -0) ? 0 : rounded;
        };

        var aRounded = roundOneDecimal(aNum);
        var isDegenerate = Math.abs(aRounded) < 1e-9;
        var vertexNotice = "";
        var zerosNotice = "";

        var axisShape = null;
        if (vertexToggleActive && !isDegenerate) {
            var xv = -bNum / (2 * aNum);
            var yv = aNum * xv * xv + bNum * xv + cNum;
            if (isFinite(xv) && isFinite(yv)) {
                vertexTrace.x = [xv];
                vertexTrace.y = [yv];
                axisShape = {
                    type: "line",
                    x0: xv,
                    x1: xv,
                    xref: "x",
                    yref: "paper",
                    y0: 0,
                    y1: 1,
                    line: {color: "#888888", width: 1, dash: "dash"},
                };
            } else {
                vertexTrace.x = [];
                vertexTrace.y = [];
            }
        } else {
            vertexTrace.x = [];
            vertexTrace.y = [];
            if (vertexToggleActive && isDegenerate) {
                vertexNotice = "a = 0 → vertex/axis hidden (linear case).";
            }
        }

        if (!axisShape) {
            vertexTrace.x = vertexTrace.x || [];
            vertexTrace.y = vertexTrace.y || [];
        }

        if (zerosToggleActive) {
            const DISC_EPS = 1e-10;
            var roots = [];
            if (!isDegenerate) {
                var disc = bNum * bNum - 4 * aNum * cNum;
                if (disc > DISC_EPS) {
                    var sqrtDisc = Math.sqrt(disc);
                    roots.push((-bNum - sqrtDisc) / (2 * aNum));
                    roots.push((-bNum + sqrtDisc) / (2 * aNum));
                } else if (Math.abs(disc) <= DISC_EPS) {
                    roots.push(-bNum / (2 * aNum));
                } else {
                    zerosNotice = "No real zeros";
                }
            } else {
                var bRounded = roundOneDecimal(bNum);
                if (Math.abs(bRounded) >= 1e-9) {
                    roots.push(-cNum / bNum);
                } else {
                    var cRounded = roundOneDecimal(cNum);
                    if (Math.abs(cRounded) < 1e-9) {
                        zerosNotice = "All x are zeros (degenerate y = 0).";
                    } else {
                        zerosNotice = "No real zeros";
                    }
                }
            }

            var finiteRoots = roots.filter((val) => isFinite(val));
            zerosTrace.x = finiteRoots;
            zerosTrace.y = finiteRoots.map(() => 0);
            if (!finiteRoots.length && zerosNotice === "") {
                zerosNotice = "No real zeros";
            } else if (finiteRoots.length && zerosNotice !== "All x are zeros (degenerate y = 0).") {
                zerosNotice = "";
            }
        } else {
            zerosTrace.x = [];
            zerosTrace.y = [];
        }

        newData[1] = vertexTrace;
        newData[2] = zerosTrace;
        var newFigure = Object.assign({}, figure);
        newFigure.data = newData;

        var newLayout = Object.assign({}, figure.layout || {});
        newLayout.uirevision = uiRevision;
        newLayout.shapes = axisShape ? [axisShape] : [];
        newFigure.layout = newLayout;

        if (!zerosToggleActive) {
            zerosNotice = "";
        }

        return [newFigure, vertexNotice, zerosNotice];
    }
    """,
    [
        Output("graph-main", "figure"),
        Output("vertex-notice", "children"),
        Output("zeros-notice", "children"),
    ],
    [
        Input("slider-a", "value"),
        Input("slider-b", "value"),
        Input("slider-c", "value"),
        Input("input-a", "value"),
        Input("input-b", "value"),
        Input("input-c", "value"),
        Input("toggle-vertex", "value"),
        Input("toggle-zeros", "value"),
        Input("store-ui", "data"),
    ],
    [
        State("graph-main", "figure"),
        State("store-x", "data"),
    ],
)


app.clientside_callback(
    r"""
    (function() {
        const toOneDecimal = (value) => {
            const num = Number(value);
            if (!isFinite(num)) {
                return 0;
            }
            const rounded = Math.round(num * 10) / 10;
            return Object.is(rounded, -0) ? 0 : rounded;
        };

        const formatNumber = (value) => {
            const rounded = toOneDecimal(value);
            const fixed = rounded.toFixed(1);
            return fixed.endsWith(".0") ? fixed.slice(0, -2) : fixed;
        };

        const isValidParam = (param) => param === "a" || param === "b" || param === "c";

        const coreForParam = (param, absCoeff) => {
            if (param === "a") {
                const symbol = "x²";
                return Math.abs(absCoeff - 1) < 1e-9 ? symbol : `${formatNumber(absCoeff)}${symbol}`;
            }
            if (param === "b") {
                const symbol = "x";
                return Math.abs(absCoeff - 1) < 1e-9 ? symbol : `${formatNumber(absCoeff)}${symbol}`;
            }
            return formatNumber(absCoeff);
        };

        return function(aValue, bValue, cValue, sourceStore) {
            const coeffs = {
                a: toOneDecimal(aValue ?? 0),
                b: toOneDecimal(bValue ?? 0),
                c: toOneDecimal(cValue ?? 0),
            };

            const highlightParam =
                sourceStore && typeof sourceStore === "object" && isValidParam(sourceStore.param)
                    ? sourceStore.param
                    : null;

            const termOrder = ["a", "b", "c"];
            const terms = [];
            termOrder.forEach((param) => {
                const coeff = coeffs[param];
                if (Math.abs(coeff) < 1e-9) {
                    return;
                }
                const absVal = Math.abs(coeff);
                terms.push({
                    param,
                    sign: coeff < 0 ? "-" : "+",
                    core: coreForParam(param, absVal),
                });
            });

            let rhs;
            if (!terms.length) {
                rhs = "0";
            } else {
                const applyHighlight = (chunk, shouldHighlight) => {
                    if (!shouldHighlight) {
                        return chunk;
                    }
                    if (chunk.startsWith(" ")) {
                        return ` **${chunk}**`;
                    }
                    return `**${chunk}**`;
                };

                const pieces = terms.map((term, idx) => {
                    if (idx === 0) {
                        const chunk = term.sign === "-" ? `-${term.core}` : term.core;
                        return applyHighlight(chunk, term.param === highlightParam);
                    }
                    const chunk = `${term.sign === "-" ? " - " : " + "}${term.core}`;
                    return applyHighlight(chunk, term.param === highlightParam);
                });
                rhs = pieces.join("");
            }

            return `y = ${rhs}`;
        };
    })()
    """,
    Output("equation-view", "children"),
    [
        Input("slider-a", "value"),
        Input("slider-b", "value"),
        Input("slider-c", "value"),
        Input("store-source", "data"),
    ],
)


app.clientside_callback(
    r"""
    (function() {
        const DEFAULT_A = 1.0;
        const NOISE_THRESHOLD = 0.05;
        const WIDTH_THRESHOLD = 0.1;
        const TIP_DURATION_MS = 2500;
        const BASE_STYLE = Object.freeze({
            minHeight: "1.5em",
            fontSize: "0.9rem",
            color: "#444444",
            marginTop: "4px",
            transition: "opacity 0.2s ease-in-out",
        });
        const hiddenStyle = Object.assign({}, BASE_STYLE, {
            opacity: 0,
            visibility: "hidden",
        });
        const visibleStyle = Object.assign({}, BASE_STYLE, {
            opacity: 1,
            visibility: "visible",
        });
        const toOneDecimal = (value, fallback = DEFAULT_A) => {
            const num = Number(value);
            const base = isFinite(num) ? num : fallback;
            const rounded = Math.round(base * 10) / 10;
            return Object.is(rounded, -0) ? 0 : rounded;
        };
        const ensureStore = (raw) => {
            if (!raw || typeof raw !== "object") {
                return {
                    a_prev: DEFAULT_A,
                    visible_until: null,
                    last_rule: null,
                };
            }
            return {
                a_prev:
                    typeof raw.a_prev === "number"
                        ? toOneDecimal(raw.a_prev)
                        : DEFAULT_A,
                visible_until:
                    typeof raw.visible_until === "number" ? raw.visible_until : null,
                last_rule: typeof raw.last_rule === "string" ? raw.last_rule : null,
            };
        };
        const orientationHint = (value) => {
            if (Math.abs(value) < 1e-9) {
                return "";
            }
            return value < 0 ? " Opens downward." : " Opens upward.";
        };
        const pickRule = (previous, current) => {
            if (Math.abs(current) < 1e-9) {
                return {
                    message: "a = 0: parabola collapses to a line; vertex/axis hidden.",
                    rule: "degenerate",
                };
            }
            if (Math.abs(previous) >= 1e-9 && previous * current < 0) {
                return {
                    message:
                        current > 0
                            ? "a changed sign → opens upward."
                            : "a changed sign → opens downward.",
                    rule: "sign_flip",
                };
            }
            const absPrev = Math.abs(previous);
            const absCurr = Math.abs(current);
            if (absCurr - absPrev >= WIDTH_THRESHOLD - 1e-9) {
                return {
                    message:
                        "Increasing |a| → parabola narrower." + orientationHint(current),
                    rule: "narrower",
                };
            }
            if (absPrev - absCurr >= WIDTH_THRESHOLD - 1e-9) {
                return {
                    message:
                        "Decreasing |a| → parabola wider." + orientationHint(current),
                    rule: "wider",
                };
            }
            return null;
        };

        return function(aValue, sourceData, intervalTicks, resetClicks, storeRaw) {
            void intervalTicks;
            const ctx =
                (window.dash_clientside && window.dash_clientside.callback_context) || {
                    triggered: [],
                };
            const triggeredIds = ctx.triggered.map((entry) => entry.prop_id);
            const intervalOnly =
                triggeredIds.length === 1 &&
                triggeredIds[0].startsWith("interval-verbal-a.");
            const resetTriggered = triggeredIds.some((id) => id.startsWith("btn-reset."));
            const storeData = ensureStore(storeRaw);
            const updatedStore = Object.assign({}, storeData);
            let storeChanged = false;
            const NO_UPDATE =
                (window.dash_clientside && window.dash_clientside.no_update) || null;
            let tipText = NO_UPDATE;
            let tipStyle = NO_UPDATE;
            let intervalDisabled =
                typeof storeData.visible_until === "number" ? false : true;
            const setStoreValue = (key, value) => {
                if (updatedStore[key] !== value) {
                    updatedStore[key] = value;
                    storeChanged = true;
                }
            };
            const hideTip = () => {
                tipText = "";
                tipStyle = hiddenStyle;
                setStoreValue("visible_until", null);
                setStoreValue("last_rule", null);
            };

            if (resetTriggered) {
                setStoreValue("a_prev", DEFAULT_A);
                hideTip();
                intervalDisabled = true;
                return [
                    storeChanged ? updatedStore : storeData,
                    tipText,
                    tipStyle,
                    intervalDisabled,
                ];
            }

            if (intervalOnly) {
                if (
                    typeof storeData.visible_until === "number" &&
                    Date.now() >= storeData.visible_until
                ) {
                    hideTip();
                    intervalDisabled = true;
                } else {
                    intervalDisabled =
                        typeof storeData.visible_until === "number" ? false : true;
                }
                return [
                    storeChanged ? updatedStore : storeData,
                    tipText,
                    tipStyle,
                    intervalDisabled,
                ];
            }

            const roundedA = toOneDecimal(
                isFinite(Number(aValue)) ? aValue : storeData.a_prev
            );
            const prevA =
                typeof storeData.a_prev === "number" ? storeData.a_prev : roundedA;
            const lastParam = sourceData && sourceData.param;
            const isAEvent = lastParam === "a";
            let messageData = null;

            if (
                isAEvent &&
                Math.abs(roundedA - prevA) >= NOISE_THRESHOLD - 1e-9
            ) {
                messageData = pickRule(prevA, roundedA);
            }

            setStoreValue("a_prev", roundedA);

            if (messageData) {
                tipText = messageData.message;
                tipStyle = visibleStyle;
                setStoreValue("visible_until", Date.now() + TIP_DURATION_MS);
                setStoreValue("last_rule", messageData.rule);
                intervalDisabled = false;
            } else {
                intervalDisabled =
                    typeof updatedStore.visible_until === "number" ? false : true;
            }

            return [
                storeChanged ? updatedStore : storeData,
                tipText,
                tipStyle,
                intervalDisabled,
            ];
        };
    })()
    """,
    [
        Output("store-a-prev", "data"),
        Output("verbal-a-tip", "children"),
        Output("verbal-a-tip", "style"),
        Output("interval-verbal-a", "disabled"),
    ],
    [
        Input("slider-a", "value"),
        Input("store-source", "data"),
        Input("interval-verbal-a", "n_intervals"),
        Input("btn-reset", "n_clicks"),
    ],
    State("store-a-prev", "data"),
)


@app.callback(
    Output("store-log-sink", "data", allow_duplicate=True),
    [
        Input("slider-a", "value"),
        Input("slider-b", "value"),
        Input("slider-c", "value"),
    ],
    [
        State("store-session", "data"),
        State("store-consent", "data"),
        State("store-source", "data"),
        State("store-ui", "data"),
        State("store-log-sink", "data"),
    ],
    prevent_initial_call=True,
)
def _log_slider_activity(a_value, b_value, c_value, session_data, consent_data, source_store, ui_store_data, log_store_data):
    a_num = _coerce_float(a_value)
    b_num = _coerce_float(b_value)
    c_num = _coerce_float(c_value)
    if a_num is None or b_num is None or c_num is None:
        return dash.no_update
    if not _is_logging_allowed(consent_data):
        return dash.no_update

    if not isinstance(source_store, dict):
        return dash.no_update
    param_name = source_store.get("param")
    if param_name not in {"a", "b", "c"}:
        return dash.no_update

    session_id = _get_session_id(session_data)
    current_values = {"a": a_num, "b": b_num, "c": c_num}
    new_value = current_values.get(param_name)
    if new_value is None:
        return dash.no_update

    try:
        meta_value = float(source_store.get("value"))
    except (TypeError, ValueError):
        meta_value = None
    if meta_value is None or abs(meta_value - new_value) > 1e-6:
        return dash.no_update

    cache = _get_param_cache(session_id)
    old_value = cache.get(param_name)
    source = _extract_source_label(source_store)
    uirevision = _resolve_uirevision_value(ui_store_data)
    record = _base_log_record(
        session_id,
        event="param_change",
        param_name=param_name,
        old_value=old_value,
        new_value=new_value,
        source=source,
        uirevision=uirevision,
        viewport=None,
    )
    _log_with_throttle(session_id, record)
    cache[param_name] = new_value
    summary = _format_preview_message(record)
    return _append_preview_log(log_store_data, summary)


@app.callback(
    Output("store-log-sink", "data", allow_duplicate=True),
    [
        Input("toggle-vertex", "value"),
        Input("toggle-zeros", "value"),
    ],
    [
        State("store-session", "data"),
        State("store-consent", "data"),
        State("store-ui", "data"),
        State("store-log-sink", "data"),
    ],
    prevent_initial_call=True,
)
def _log_overlay_toggle(vertex_value, zeros_value, session_data, consent_data, ui_store_data, log_store_data):
    if not _is_logging_allowed(consent_data):
        return dash.no_update
    session_id = _get_session_id(session_data)
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    overlay_key = None
    new_enabled = None
    if trigger_id == "toggle-vertex":
        overlay_key = "vertex_axis"
        new_enabled = _is_toggle_enabled(vertex_value)
    elif trigger_id == "toggle-zeros":
        overlay_key = "zeros"
        new_enabled = _is_toggle_enabled(zeros_value)
    else:
        return dash.no_update

    overlay_state = _get_overlay_state(session_id)
    old_value = overlay_state.get(overlay_key)
    if old_value == new_enabled:
        return dash.no_update

    overlay_state[overlay_key] = new_enabled
    uirevision = _resolve_uirevision_value(ui_store_data)
    record = _base_log_record(
        session_id,
        event="overlay_toggle",
        param_name=None,
        old_value=old_value,
        new_value=new_enabled,
        source="toggle",
        uirevision=uirevision,
        viewport=None,
        extras={
            "overlay": overlay_key,
            "enabled": bool(new_enabled),
        },
    )
    _write_log_record(session_id, record)
    summary = _format_preview_message(record)
    return _append_preview_log(log_store_data, summary)


@app.callback(
    [
        Output("download-jsonl", "data"),
        Output("store-log-sink", "data", allow_duplicate=True),
    ],
    Input("btn-download-jsonl", "n_clicks"),
    [
        State("store-session", "data"),
        State("store-consent", "data"),
        State("store-ui", "data"),
        State("store-log-sink", "data"),
    ],
    prevent_initial_call=True,
)
def _handle_download_jsonl(n_clicks, session_data, consent_data, ui_store_data, log_store_data):
    if not n_clicks or not _is_logging_allowed(consent_data):
        return dash.no_update, dash.no_update
    session_id = _get_session_id(session_data)
    path = _get_session_log_path(session_id)
    if not path.exists():
        return dash.no_update, dash.no_update

    record = _base_log_record(
        session_id,
        event="export",
        source="button",
        uirevision=_resolve_uirevision_value(ui_store_data),
        viewport=None,
        extras={"export_type": "jsonl"},
    )
    _write_log_record(session_id, record)
    summary = _format_preview_message(record)
    return dcc.send_file(str(path)), _append_preview_log(log_store_data, summary)


@app.callback(
    [
        Output("download-csv", "data"),
        Output("store-log-sink", "data", allow_duplicate=True),
    ],
    Input("btn-download-csv", "n_clicks"),
    [
        State("store-session", "data"),
        State("store-consent", "data"),
        State("store-ui", "data"),
        State("store-log-sink", "data"),
    ],
    prevent_initial_call=True,
)
def _handle_download_csv(n_clicks, session_data, consent_data, ui_store_data, log_store_data):
    if not n_clicks or not _is_logging_allowed(consent_data):
        return dash.no_update, dash.no_update
    session_id = _get_session_id(session_data)
    records = _read_session_log_records(session_id)
    csv_content = _build_csv_content(records)
    if not csv_content:
        return dash.no_update, dash.no_update

    record = _base_log_record(
        session_id,
        event="export",
        source="button",
        uirevision=_resolve_uirevision_value(ui_store_data),
        viewport=None,
        extras={"export_type": "csv"},
    )
    _write_log_record(session_id, record)
    summary = _format_preview_message(record)
    filename = f"session_{_safe_session_id(session_id)}.csv"
    return dcc.send_string(csv_content, filename=filename), _append_preview_log(log_store_data, summary)


@app.callback(
    [
        Output("slider-a", "value", allow_duplicate=True),
        Output("slider-b", "value", allow_duplicate=True),
        Output("slider-c", "value", allow_duplicate=True),
        Output("input-a", "value", allow_duplicate=True),
        Output("input-b", "value", allow_duplicate=True),
        Output("input-c", "value", allow_duplicate=True),
        Output("store-ui", "data"),
        Output("store-log-sink", "data", allow_duplicate=True),
    ],
    Input("btn-reset", "n_clicks"),
    State("store-ui", "data"),
    State("store-session", "data"),
    State("store-consent", "data"),
    State("store-log-sink", "data"),
    prevent_initial_call=True,
)
def _handle_reset(n_clicks, ui_store_data, session_data, consent_data, log_store_data):
    if not n_clicks:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    updated_ui_store = _bump_uirevision_store(ui_store_data)
    session_id = _get_session_id(session_data)
    _SESSION_PARAM_CACHE[session_id] = dict(_DEFAULT_PARAMS)
    log_update = dash.no_update
    if _is_logging_allowed(consent_data):
        record = _base_log_record(
            session_id,
            event="reset",
            source="button",
            uirevision=_resolve_uirevision_value(updated_ui_store),
            viewport=None,
        )
        _write_log_record(session_id, record)
        log_update = _append_preview_log(log_store_data, _format_preview_message(record))
    return (
        _DEFAULT_PARAMS["a"],
        _DEFAULT_PARAMS["b"],
        _DEFAULT_PARAMS["c"],
        _DEFAULT_PARAMS["a"],
        _DEFAULT_PARAMS["b"],
        _DEFAULT_PARAMS["c"],
        updated_ui_store,
        log_update,
    )


@app.callback(
    Output("log-display", "children"),
    [
        Input("store-log-sink", "data"),
        Input("store-consent", "data"),
    ],
)
def _render_log_display(log_entries, consent_data):
    if not _is_logging_allowed(consent_data):
        return "Recent logs hidden (consent required)."
    if not isinstance(log_entries, list):
        log_entries = []
    if not log_entries:
        return "Recent logs will appear here."
    lines = [f"- {entry}" for entry in reversed(log_entries)]
    return "\n".join(["Recent logs:", *lines])


if __name__ == "__main__":
    app.run_server(debug=True)
