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
_TRACE_HISTORY_CAPACITY = 10
_TRACE_STORE_DEFAULT: Dict[str, Any] = {"entries": [], "capacity": _TRACE_HISTORY_CAPACITY}

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
_DEFAULT_OVERLAY_STATE = {"vertex_axis": True, "zeros": True, "trace": True}
_OVERLAY_STATE: Dict[str, Dict[str, bool]] = {}
_REFLECTION_MAX_LEN = 1000
_REFLECTION_SOFT_MIN = 20
_REFLECTION_SAVED_MS = 2000
_DEFAULT_REFLECTION_DRAFT: Dict[str, Any] = {
    "text": "",
    "focused_ts": None,
    "first_meaningful_edit_ts": None,
    "last_edit_ts": None,
    "edit_count": 0,
    "initial_len": 0,
    "paste_flag": False,
}
_DEFAULT_REFLECTION_STATUS: Dict[str, Any] = {
    "next_seq": 1,
    "pending_seq": None,
    "submitting": False,
    "saved_until": None,
    "saved_seq": None,
    "last_ack_seq": None,
}
_LAST_REFLECTION_SEQ: Dict[str, int] = {}
_DEFAULT_ABOUT_STATE: Dict[str, Any] = {"seen": False, "open": True}
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
    "reflection_text",
    "chars",
    "words",
    "draft_total_ms",
    "idle_to_submit_ms",
    "edit_count",
    "char_delta",
    "paste_flag",
    "accidental_focus_flag",
    "t_client",
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


def _default_reflection_draft() -> Dict[str, Any]:
    return dict(_DEFAULT_REFLECTION_DRAFT)


def _default_reflection_status() -> Dict[str, Any]:
    return dict(_DEFAULT_REFLECTION_STATUS)


def _default_about_state() -> Dict[str, Any]:
    return dict(_DEFAULT_ABOUT_STATE)


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
    flat["reflection_text"] = record.get("reflection_text")
    flat["chars"] = record.get("chars")
    flat["words"] = record.get("words")
    flat["draft_total_ms"] = record.get("draft_total_ms")
    flat["idle_to_submit_ms"] = record.get("idle_to_submit_ms")
    flat["edit_count"] = record.get("edit_count")
    flat["char_delta"] = record.get("char_delta")
    flat["paste_flag"] = record.get("paste_flag")
    flat["accidental_focus_flag"] = record.get("accidental_focus_flag")
    flat["t_client"] = record.get("t_client")
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
    if event == "reflection_submit":
        chars = record.get("chars")
        try:
            chars_val = int(chars)
        except (TypeError, ValueError):
            chars_val = None
        char_text = f"{chars_val}" if chars_val is not None else "?"
        return f"reflection_submit: {char_text} chars"
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
                line=dict(color="#0072B2", width=3),
            ),
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                name="Vertex",
                marker=dict(color="#D55E00", size=10, symbol="circle", line=dict(color="#ffffff", width=1)),
                hovertemplate="Vertex<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                name="Zeros",
                marker=dict(color="#000000", size=9, symbol="x", line=dict(color="#ffffff", width=1)),
                hovertemplate="Zero<br>x=%{x:.2f}<extra></extra>",
                showlegend=False,
            ),
        ]
    )
    for _ in range(_TRACE_HISTORY_CAPACITY):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Trace",
                line=dict(color="rgba(0,114,178,0.35)", width=1.5),
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        height=560,
        margin=dict(l=36, r=16, t=32, b=32),
        xaxis=dict(
            title="x",
            showgrid=True,
            zeroline=True,
            zerolinecolor="#777777",
        ),
        yaxis=dict(
            title="y",
            showgrid=True,
            zeroline=True,
            zerolinecolor="#777777",
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
    desc_id = f"desc-{param}"
    slider_labels = {
        "a": "Coefficient a (quadratic term)",
        "b": "Coefficient b (linear term)",
        "c": "Coefficient c (constant term)",
    }
    input_labels = {
        "a": "Coefficient a (exact value)",
        "b": "Coefficient b (exact value)",
        "c": "Coefficient c (exact value)",
    }
    slider_titles = {
        "a": "Drag to change a (opening & width). Larger |a| narrows; a < 0 flips downward.",
        "b": "Drag to change b (horizontal shift). Vertex x = -b/(2a).",
        "c": "Drag to change c (vertical shift up/down).",
    }
    input_titles = {
        "a": "Type an exact a value (-5.0 to 5.0, step 0.1).",
        "b": "Type an exact b value (-10.0 to 10.0, step 0.1).",
        "c": "Type an exact c value (-10.0 to 10.0, step 0.1).",
    }
    minus_titles = {
        "a": "Decrease a by 0.1.",
        "b": "Decrease b by 0.1.",
        "c": "Decrease c by 0.1.",
    }
    plus_titles = {
        "a": "Increase a by 0.1.",
        "b": "Increase b by 0.1.",
        "c": "Increase c by 0.1.",
    }
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
                        title=slider_titles[param],
                    ),
                    html.Div(
                        dcc.Input(
                            id=input_id,
                            type="number",
                            min=cfg["min"],
                            max=cfg["max"],
                            step=cfg["step"],
                            value=_DEFAULT_PARAMS[param],
                            className="numeric-input",
                            placeholder=input_titles[param],
                            style={
                                "width": "96px",
                                "marginLeft": "12px",
                                "marginRight": "8px",
                                "height": "44px",
                            },
                        ),
                        role="group",
                        title=input_titles[param],
                        **{
                            "aria-label": input_labels[param],
                            "aria-describedby": desc_id,
                        },
                    ),
                    html.Button(
                        "-",
                        id=minus_id,
                        n_clicks=0,
                        className="a11y-target",
                        type="button",
                        style={
                            "width": "44px",
                            "height": "44px",
                            "marginRight": "4px",
                        },
                        title=minus_titles[param],
                        **{"aria-label": f"Decrease {param}"},
                    ),
                    html.Button(
                        "+",
                        id=plus_id,
                        n_clicks=0,
                        className="a11y-target",
                        type="button",
                        style={"width": "44px", "height": "44px"},
                        title=plus_titles[param],
                        **{"aria-label": f"Increase {param}"},
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
    global_styles = dcc.Markdown(
        """
<style>
.visually-hidden {
    position: absolute !important;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
button:focus-visible,
input:focus-visible,
textarea:focus-visible,
.rc-slider-handle:focus-visible,
.dash-core-components input:focus-visible {
    outline: 3px solid #ffbf47;
    outline-offset: 2px;
}
.a11y-target {
    min-width: 44px;
    min-height: 44px;
}
.numeric-input {
    text-align: right;
}
.about-info-button {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    border: 1px solid #004a7c;
    background-color: #0072b2;
    color: #ffffff;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    z-index: 1300;
}
.about-info-button:hover {
    background-color: #005b8a;
}
</style>
        """,
        dangerously_allow_html=True,
    )

    descriptions = html.Div(
        [
            html.Span(
                "Controls width and opening; larger |a| narrows; a < 0 opens downward.",
                id="desc-a",
            ),
            html.Span(
                "Moves the vertex left/right; vertex x = -b/(2a) when a != 0.",
                id="desc-b",
            ),
            html.Span(
                "Shifts the graph up/down; y-intercept at (0, c).",
                id="desc-c",
            ),
        ],
        className="visually-hidden",
        style={
            "position": "absolute",
            "width": "1px",
            "height": "1px",
            "padding": "0",
            "margin": "-1px",
            "overflow": "hidden",
            "clip": "rect(0, 0, 0, 0)",
            "whiteSpace": "nowrap",
            "border": "0",
        },
        **{"aria-hidden": "true"},
    )

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
                className="a11y-target",
                style={"marginTop": "8px"},
                title="Reset parameters and reset view",
                **{"aria-label": "Reset parameters and reset view"},
            ),
            html.Div(
                dcc.Checklist(
                    id="toggle-vertex",
                    options=[{"label": "Vertex & axis", "value": "on"}],
                    value=["on"],
                    labelStyle={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "6px",
                    },
                    inputStyle={"marginRight": "6px"},
                ),
                style={"marginTop": "16px"},
            ),
            html.Div(
                dcc.Checklist(
                    id="toggle-zeros",
                    options=[{"label": "Zeros (x-intercepts)", "value": "on"}],
                    value=["on"],
                    labelStyle={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "6px",
                    },
                    inputStyle={"marginRight": "6px"},
                ),
                style={"marginTop": "8px"},
            ),
            html.Div(
                dcc.Checklist(
                    id="toggle-trace",
                    options=[{"label": "Trace mode", "value": "on"}],
                    value=["on"],
                    labelStyle={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "6px",
                    },
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
                        role="status",
                        **{"aria-live": "polite"},
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
                        role="status",
                        **{"aria-live": "polite"},
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
                        role="status",
                        **{"aria-live": "polite"},
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
            html.Div(
                [
                    html.H3("Reflection"),
                    html.P(
                        "Capture a quick note about what you observed. "
                        "We'll rotate the prompt later, but for now focus on changes in a."
                    ),
                    dcc.Textarea(
                        id="reflect-input",
                        value="",
                        placeholder="Describe what changed when you increased a.",
                        rows=4,
                        maxLength=_REFLECTION_MAX_LEN,
                        style={
                            "width": "100%",
                            "minHeight": "120px",
                            "padding": "10px",
                            "fontSize": "1rem",
                            "borderRadius": "8px",
                            "border": "1px solid #cccccc",
                            "resize": "vertical",
                        },
                    ),
                    html.Div(
                        [
                            html.Span("0 / 1000 characters", id="reflect-char-count"),
                            html.Span(
                                "Write at least 20 characters to build a useful log.",
                                id="reflect-soft-hint",
                                style={"color": "#555555"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginTop": "6px",
                            "fontSize": "0.9rem",
                        },
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Submit Reflection",
                                id="reflect-submit",
                                n_clicks=0,
                                disabled=True,
                                className="a11y-target",
                                style={"padding": "10px 18px", "fontWeight": 600},
                                title="Submit reflection",
                                **{"aria-label": "Submit reflection"},
                            ),
                            html.Span(
                                "Saved",
                                id="reflect-saved-indicator",
                                style={
                                    "marginLeft": "12px",
                                    "fontSize": "0.9rem",
                                    "color": "#2e8b57",
                                    "opacity": 0,
                                    "transition": "opacity 0.25s ease-in-out",
                                },
                            ),
                        ],
                        style={"marginTop": "12px", "display": "flex", "alignItems": "center"},
                    ),
                    html.Div(
                        "",
                        id="reflect-error-message",
                        style={"color": "#cc3311", "fontSize": "0.9rem", "minHeight": "1.2em"},
                    ),
                ],
                style={"marginTop": "32px"},
            ),
            html.Div(
                [
                    html.H3("History"),
                    html.Div(
                        "Trace history will appear here.",
                        id="trace-history",
                        style={
                            "fontFamily": "monospace",
                            "fontSize": "0.9rem",
                            "backgroundColor": "#f7f7f7",
                            "borderRadius": "8px",
                            "padding": "12px",
                            "marginTop": "8px",
                            "minHeight": "100px",
                            "whiteSpace": "pre-line",
                        },
                    ),
                ],
                style={"marginTop": "32px"},
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
            dcc.Store(id="store-param-pending", data=None),
            dcc.Store(
                id="store-a-prev",
                data={"a_prev": _DEFAULT_PARAMS["a"], "visible_until": None, "last_rule": None},
            ),
            dcc.Store(id="store-param-commit", data=None),
            dcc.Store(id="store-trace", data=dict(_TRACE_STORE_DEFAULT)),
            dcc.Store(id="store-reflection-draft", data=_default_reflection_draft()),
            dcc.Store(id="store-reflection-status", data=_default_reflection_status()),
            dcc.Store(id="store-reflection-commit", data=None),
            dcc.Store(id="store-reflection-ack", data=None),
            dcc.Store(id="store-about", storage_type="local", data=_default_about_state()),
            dcc.Interval(id="interval-verbal-a", interval=500, n_intervals=0, disabled=True),
            dcc.Interval(id="interval-reflect-saved", interval=500, n_intervals=0, disabled=True),
            dcc.Interval(id="interval-param-commit", interval=200, n_intervals=0, disabled=True),
            global_styles,
            descriptions,
            html.Div(main_content, style={"padding": "32px"}),
            representations_section,
            html.Div("", id="live-status", className="visually-hidden", role="status", **{"aria-live": "polite"}),
            html.Div("", id="about-focus-anchor", style={"display": "none"}),
            _build_consent_modal(),
            html.Div(
                id="about-overlay",
                n_clicks=0,
                style={
                    **_MODAL_OVERLAY_BASE_STYLE,
                    "display": "none",
                    "zIndex": 1200,
                },
                **{"aria-hidden": "true"},
                children=[
                    html.Div(
                        [
                            html.H2("About Function Explorer", id="about-modal-title", style={"marginTop": "0"}),
                            html.P("We explore quadratics of the form y = ax^2 + bx + c."),
                            html.Ul(
                                [
                                    html.Li("a (quadratic term): controls opening direction and width."),
                                    html.Li("b (linear term): shifts the vertex left/right."),
                                    html.Li("c (constant term): moves the graph up or down."),
                                ],
                                id="about-modal-desc",
                                style={"paddingLeft": "20px"},
                            ),
                            html.P("Tip: drag slowly or use keyboard nudges to see how each coefficient shapes the curve."),
                            html.Button(
                                "Got it",
                                id="btn-about-dismiss",
                                n_clicks=0,
                                className="a11y-target",
                                style={"marginTop": "16px"},
                            ),
                            html.Button("", id="btn-about-esc", style={"display": "none"}),
                        ],
                        id="about-modal-panel",
                        n_clicks=0,
                        style={
                            **_MODAL_PANEL_STYLE,
                            "outline": "none",
                        },
                        role="dialog",
                        **{
                            "aria-modal": "true",
                            "aria-labelledby": "about-modal-title",
                            "aria-describedby": "about-modal-desc",
                            "tabIndex": -1,
                        },
                    )
                ],
            ),
            html.Button(
                "i",
                id="btn-about-info",
                n_clicks=0,
                className="about-info-button a11y-target",
                title="About Function Explorer",
                **{"aria-label": "Open About dialog"},
            ),
            html.Script(
                """
                (function() {
                    if (window.__aboutModalEscAttached) {
                        return;
                    }
                    window.__aboutModalEscAttached = true;
                    var panel = document.getElementById("about-modal-panel");
                    if (panel) {
                        panel.addEventListener("click", function(event) {
                            event.stopPropagation();
                        });
                    }
                    document.addEventListener("keydown", function(event) {
                        if (event.key === "Escape") {
                            var overlay = document.getElementById("about-overlay");
                            if (!overlay) {
                                return;
                            }
                            var visible = overlay.style.display && overlay.style.display !== "none";
                            if (visible) {
                                var escButton = document.getElementById("btn-about-esc");
                                if (escButton) {
                                    escButton.click();
                                }
                            }
                        }
                    });
                })();
                """,
                id="about-modal-script",
            ),
            html.Script(
                """
                (function() {
                    if (window.__reflectShortcutAttached) {
                        return;
                    }
                    window.__reflectShortcutAttached = true;
                    document.addEventListener("keydown", function(event) {
                        if (!(event.metaKey || event.ctrlKey) || event.key !== "Enter") {
                            return;
                        }
                        const active = document.activeElement;
                        const textarea = document.getElementById("reflect-input");
                        if (!textarea || active !== textarea) {
                            return;
                        }
                        const button = document.getElementById("reflect-submit");
                        if (!button || button.disabled) {
                            return;
                        }
                        event.preventDefault();
                        button.click();
                    });
                })();
                """,
                id="reflect-shortcut-script",
            ),
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


@app.callback(
    Output("store-about", "data"),
    Input("btn-about-info", "n_clicks"),
    Input("btn-about-dismiss", "n_clicks"),
    Input("about-overlay", "n_clicks"),
    Input("btn-about-esc", "n_clicks"),
    State("store-about", "data"),
    prevent_initial_call=True,
)
def _handle_about_modal(info_clicks, dismiss_clicks, overlay_clicks, esc_clicks, about_data):
    data = about_data if isinstance(about_data, dict) else _default_about_state()
    data = dict(data)
    ctx = dash.callback_context
    if not ctx.triggered:
        return data
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "btn-about-info":
        data["open"] = True
        data["seen"] = True
    elif trigger_id in {"btn-about-dismiss", "btn-about-esc"}:
        data["open"] = False
        data["seen"] = True
    elif trigger_id == "about-overlay":
        if data.get("open"):
            data["open"] = False
            data["seen"] = True
    return data


@app.callback(
    Output("about-overlay", "style"),
    Input("store-about", "data"),
)
def _sync_about_modal_style(about_data):
    data = about_data if isinstance(about_data, dict) else _default_about_state()
    style = dict(_MODAL_OVERLAY_BASE_STYLE)
    style.update({"zIndex": 1200, "padding": "16px"})
    style["display"] = "flex" if data.get("open") else "none"
    return style


app.clientside_callback(
    """
    function(state) {
        const overlay = document.getElementById("about-overlay");
        const dismissButton = document.getElementById("btn-about-dismiss");
        const infoButton = document.getElementById("btn-about-info");
        if (!state || typeof state !== "object") {
            if (overlay) {
                overlay.setAttribute("aria-hidden", "true");
            }
            return "";
        }
        const isOpen = Boolean(state.open);
        if (overlay) {
            overlay.setAttribute("aria-hidden", isOpen ? "false" : "true");
        }
        if (isOpen) {
            setTimeout(function() {
                if (dismissButton) {
                    dismissButton.focus();
                }
            }, 0);
        } else {
            setTimeout(function() {
                if (infoButton) {
                    infoButton.focus();
                }
            }, 0);
        }
        return "";
    }
    """,
    Output("about-focus-anchor", "children"),
    Input("store-about", "data"),
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
        const defaultStatus = () => ({
            next_seq: 1,
            pending_seq: null,
            submitting: false,
            saved_until: null,
            saved_seq: null,
            last_ack_seq: null,
        });
        return function(ackData, statusRaw) {
            const ack = ackData && typeof ackData === "object" ? ackData : null;
            if (!ack || typeof ack.seq !== "number") {
                return window.dash_clientside.no_update;
            }
            const status = statusRaw && typeof statusRaw === "object" ? Object.assign(defaultStatus(), statusRaw) : defaultStatus();
            const alreadyHandled =
                typeof status.last_ack_seq === "number" && status.last_ack_seq >= ack.seq;
            if (alreadyHandled) {
                return window.dash_clientside.no_update;
            }
            const next = Object.assign({}, status, {
                last_ack_seq: ack.seq,
                submitting: false,
            });
            const shouldClearPending =
                typeof status.pending_seq === "number" && status.pending_seq === ack.seq;
            if (shouldClearPending) {
                next.pending_seq = null;
            }
            if (ack.status === "ok") {
                next.saved_until = Date.now() + %d;
                next.saved_seq = ack.seq;
                if (typeof next.next_seq !== "number" || next.next_seq <= ack.seq) {
                    next.next_seq = ack.seq + 1;
                }
            } else {
                next.saved_until = null;
                next.saved_seq = null;
            }
            return next;
        };
    })()
    """ % _REFLECTION_SAVED_MS,
    Output("store-reflection-status", "data", allow_duplicate=True),
    Input("store-reflection-ack", "data"),
    State("store-reflection-status", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    (function() {
        const MAX_LEN = %d;
        const ensureDraft = (raw) => {
            if (!raw || typeof raw !== "object") {
                return {
                    text: "",
                    focused_ts: null,
                    first_meaningful_edit_ts: null,
                    last_edit_ts: null,
                    edit_count: 0,
                    initial_len: 0,
                    paste_flag: false,
                };
            }
            return raw;
        };
        const ensureStatus = (raw) => {
            if (!raw || typeof raw !== "object") {
                return {
                    next_seq: 1,
                    pending_seq: null,
                    submitting: false,
                    saved_until: null,
                    saved_seq: null,
                    last_ack_seq: null,
                };
            }
            const status = Object.assign(
                {
                    next_seq: 1,
                    pending_seq: null,
                    submitting: false,
                    saved_until: null,
                    saved_seq: null,
                    last_ack_seq: null,
                },
                raw
            );
            if (typeof status.next_seq !== "number" || !isFinite(status.next_seq)) {
                status.next_seq = 1;
            }
            return status;
        };
        const countWords = (text) => {
            if (!text) {
                return 0;
            }
            const tokens = text.trim().split(/\\s+/).filter(Boolean);
            return tokens.length;
        };

        return function(nClicks, draftRaw, consentData, statusRaw, value) {
            const ctx = window.dash_clientside && window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || !ctx.triggered.length) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            const triggered = ctx.triggered[0].prop_id;
            if (!triggered.startsWith("reflect-submit.")) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            const consentGranted = Boolean(consentData && consentData.granted);
            if (!consentGranted) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            const draft = ensureDraft(draftRaw);
            const status = ensureStatus(statusRaw);
            if (status.submitting) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            const text = typeof value === "string" ? value : "";
            const trimmed = text.trim();
            if (!trimmed) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            if (trimmed.length > MAX_LEN) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            const now = Date.now();
            const seq = status.next_seq || 1;
            const words = countWords(trimmed);
            const draftStart = typeof draft.first_meaningful_edit_ts === "number" ? draft.first_meaningful_edit_ts : null;
            const draftTotal = draftStart ? now - draftStart : null;
            const lastEdit = typeof draft.last_edit_ts === "number" ? draft.last_edit_ts : null;
            const idleToSubmit = lastEdit ? now - lastEdit : null;
            const focusedTs = typeof draft.focused_ts === "number" ? draft.focused_ts : null;
            const accidentalFocus =
                focusedTs !== null &&
                draftStart !== null &&
                draftStart - focusedTs > 2000;
            const initialLen = typeof draft.initial_len === "number" ? draft.initial_len : 0;
            const charDelta = trimmed.length - initialLen;
            const commit = {
                seq: seq,
                t_client: now,
                reflection_text: trimmed,
                chars: trimmed.length,
                words: words,
                draft_total_ms: draftTotal,
                idle_to_submit_ms: idleToSubmit,
                edit_count: draft.edit_count || 0,
                char_delta: charDelta,
                paste_flag: Boolean(draft.paste_flag),
                accidental_focus_flag: Boolean(accidentalFocus),
            };
            const nextStatus = Object.assign({}, status, {
                pending_seq: seq,
                submitting: true,
                saved_until: null,
                saved_seq: null,
                last_ack_seq: status.last_ack_seq || null,
                next_seq: seq + 1,
            });
            return [commit, nextStatus];
        };
    })()
    """ % _REFLECTION_MAX_LEN,
    [
        Output("store-reflection-commit", "data"),
        Output("store-reflection-status", "data", allow_duplicate=True),
    ],
    Input("reflect-submit", "n_clicks"),
    [
        State("store-reflection-draft", "data"),
        State("store-consent", "data"),
        State("store-reflection-status", "data"),
        State("reflect-input", "value"),
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

        if (!window.__paramSyncState || !window.__paramSyncState.slider) {
            window.__paramSyncState = {slider: {a: DEFAULTS.a, b: DEFAULTS.b, c: DEFAULTS.c}};
        } else {
            const sliderState = window.__paramSyncState.slider;
            ["a", "b", "c"].forEach((key) => {
                if (typeof sliderState[key] !== "number" || !isFinite(sliderState[key])) {
                    sliderState[key] = DEFAULTS[key];
                }
            });
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
    (function() {
        const DEFAULTS = {a: 1.0, b: 0.0, c: 0.0};
        const PARAMS = ["a", "b", "c"];
        const DELAY_MS = 220;
        const toNumber = (value, fallback) => {
            const num = Number(value);
            return isFinite(num) ? num : fallback;
        };
        if (!window.__paramCommitState) {
            window.__paramCommitState = {
                nextSeq: 1,
                values: {a: DEFAULTS.a, b: DEFAULTS.b, c: DEFAULTS.c},
            };
        }
        const ensurePending = (raw) => {
            if (!raw || typeof raw !== "object") {
                return null;
            }
            if (!raw.commit || typeof raw.ready_at !== "number") {
                return null;
            }
            return raw;
        };
        return function(sourceStore, intervalTicks, sliderA, sliderB, sliderC, pendingRaw) {
            void intervalTicks;
            const ctx =
                (window.dash_clientside && window.dash_clientside.callback_context) || {
                    triggered: [],
                };
            const triggered = ctx.triggered.length ? ctx.triggered[0].prop_id : "";
            const seqState = window.__paramCommitState;
            let pending = ensurePending(pendingRaw);
            const noUpdate = window.dash_clientside.no_update;

            const flushCommit = (commitObj) => {
                if (!commitObj) {
                    return [pending || null, pending ? false : true, noUpdate];
                }
                const seqValue =
                    typeof commitObj.seq === "number" ? commitObj.seq : seqState.nextSeq || 1;
                commitObj.seq = seqValue;
                if (!seqState.values) {
                    seqState.values = {a: DEFAULTS.a, b: DEFAULTS.b, c: DEFAULTS.c};
                }
                if (commitObj.param) {
                    seqState.values[commitObj.param] = commitObj.new_value;
                }
                seqState.nextSeq = seqValue + 1;
                return [null, true, commitObj];
            };

            if (!triggered) {
                return [pending || null, pending ? false : true, noUpdate];
            }

            if (triggered === "interval-param-commit.n_intervals") {
                if (!pending) {
                    return [null, true, noUpdate];
                }
                if (Date.now() < pending.ready_at) {
                    return [pending, false, noUpdate];
                }
                const commitObj = pending.commit;
                pending = null;
                return flushCommit(commitObj);
            }

            if (triggered === "store-source.data") {
                if (!sourceStore || typeof sourceStore !== "object") {
                    return [pending || null, pending ? false : true, noUpdate];
                }
                const param = sourceStore.param;
                if (!param || PARAMS.indexOf(param) === -1) {
                    return [pending || null, pending ? false : true, noUpdate];
                }
                const sliderValues = {
                    a: toNumber(sliderA, seqState.values.a),
                    b: toNumber(sliderB, seqState.values.b),
                    c: toNumber(sliderC, seqState.values.c),
                };
                const newValue = sliderValues[param];
                if (!isFinite(newValue)) {
                    return [pending || null, pending ? false : true, noUpdate];
                }
                if (sourceStore.value !== undefined && sourceStore.value !== null) {
                    const meta = Number(sourceStore.value);
                    if (isFinite(meta) && Math.abs(meta - newValue) > 1e-6) {
                        return [pending || null, pending ? false : true, noUpdate];
                    }
                }
                const oldValue = seqState.values[param];
                if (isFinite(oldValue) && Math.abs(oldValue - newValue) < 1e-9) {
                    return [pending || null, pending ? false : true, noUpdate];
                }
                const commit = {
                    seq: seqState.nextSeq || 1,
                    param: param,
                    old_value: oldValue,
                    new_value: newValue,
                    source: sourceStore.source || "slider",
                    t_client:
                        typeof sourceStore.t === "number" && isFinite(sourceStore.t)
                            ? sourceStore.t
                            : Date.now(),
                    a: sliderValues.a,
                    b: sliderValues.b,
                    c: sliderValues.c,
                };
                const delay =
                    sourceStore.source === "slider" || sourceStore.source === "input_drag"
                        ? DELAY_MS
                        : 0;
                if (delay === 0) {
                    return flushCommit(commit);
                }
                pending = {
                    commit: commit,
                    ready_at: Date.now() + DELAY_MS,
                };
                return [pending, false, noUpdate];
            }

            return [pending || null, pending ? false : true, noUpdate];
        };
    })()
    """,
    [
        Output("store-param-pending", "data"),
        Output("interval-param-commit", "disabled"),
        Output("store-param-commit", "data"),
    ],
    [
        Input("store-source", "data"),
        Input("interval-param-commit", "n_intervals"),
    ],
    [
        State("slider-a", "value"),
        State("slider-b", "value"),
        State("slider-c", "value"),
        State("store-param-pending", "data"),
    ],
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    (function() {
        const CAPACITY = 10;
        const DEFAULTS = {a: 1.0, b: 0.0, c: 0.0};
        const ensureStore = (raw) => {
            if (!raw || typeof raw !== "object") {
                return {entries: [], capacity: CAPACITY};
            }
            const entries = Array.isArray(raw.entries) ? raw.entries.slice(0, CAPACITY) : [];
            return {entries, capacity: CAPACITY};
        };
        const isToggleOn = (value) => {
            if (Array.isArray(value)) {
                return value.indexOf("on") !== -1;
            }
            if (value === null || value === undefined) {
                return false;
            }
            return Boolean(value);
        };
        const roundOne = (value) => {
            const num = Number(value);
            if (!isFinite(num)) {
                return 0;
            }
            const rounded = Math.round(num * 10) / 10;
            return Object.is(rounded, -0) ? 0 : rounded;
        };
        const computeYs = (a, b, c, xs) => {
            if (!Array.isArray(xs)) {
                return [];
            }
            const size = xs.length;
            const ys = new Array(size);
            for (let i = 0; i < size; i += 1) {
                const x = Number(xs[i]);
                if (!isFinite(x)) {
                    ys[i] = null;
                } else {
                    ys[i] = a * x * x + b * x + c;
                }
            }
            return ys;
        };
        const buildSnapshot = (commit, xs) => {
            if (!commit || typeof commit !== "object" || !Array.isArray(xs)) {
                return null;
            }
            const a = Number(commit.a);
            const b = Number(commit.b);
            const c = Number(commit.c);
            if (!isFinite(a) || !isFinite(b) || !isFinite(c)) {
                return null;
            }
            return {
                seq: typeof commit.seq === "number" ? commit.seq : Date.now(),
                a: roundOne(a),
                b: roundOne(b),
                c: roundOne(c),
                y: computeYs(a, b, c, xs),
                t_client:
                    typeof commit.t_client === "number" && isFinite(commit.t_client)
                        ? commit.t_client
                        : Date.now(),
                param: commit.param || "?",
                old_value: commit.old_value,
                new_value: commit.new_value,
                source: commit.source || "slider",
            };
        };
        const buildBaseline = (xs) => {
            return buildSnapshot(
                {
                    seq: 0,
                    a: DEFAULTS.a,
                    b: DEFAULTS.b,
                    c: DEFAULTS.c,
                    param: "reset",
                    old_value: null,
                    new_value: null,
                    source: "reset",
                    t_client: Date.now(),
                },
                xs
            );
        };
        return function(commitData, toggleValue, resetClicks, consentData, traceRaw, xs) {
            const ctx =
                (window.dash_clientside && window.dash_clientside.callback_context) || {
                    triggered: [],
                };
            const triggered = ctx.triggered.length ? ctx.triggered[0].prop_id : "";
            const store = ensureStore(traceRaw);
            const consentGranted = Boolean(consentData && consentData.granted);
            const traceEnabled = consentGranted && isToggleOn(toggleValue);
            if (!consentGranted || !traceEnabled) {
                if (store.entries.length) {
                    return {entries: [], capacity: CAPACITY};
                }
                return window.dash_clientside.no_update;
            }
            if (triggered.startsWith("toggle-trace.")) {
                return {entries: [], capacity: CAPACITY};
            }
            if (triggered.startsWith("store-consent.")) {
                return {entries: [], capacity: CAPACITY};
            }
            if (triggered.startsWith("btn-reset.")) {
                const baseline = Array.isArray(xs) ? buildBaseline(xs) : null;
                if (baseline) {
                    return {entries: [baseline], capacity: CAPACITY};
                }
                return {entries: [], capacity: CAPACITY};
            }
            if (triggered.startsWith("store-param-commit.")) {
                if (!commitData || typeof commitData !== "object" || !Array.isArray(xs)) {
                    return window.dash_clientside.no_update;
                }
                const snapshot = buildSnapshot(commitData, xs);
                if (!snapshot) {
                    return window.dash_clientside.no_update;
                }
                const entries = [snapshot].concat(store.entries);
                return {entries: entries.slice(0, CAPACITY), capacity: CAPACITY};
            }
            return window.dash_clientside.no_update;
        };
    })()
    """,
    Output("store-trace", "data"),
    [
        Input("store-param-commit", "data"),
        Input("toggle-trace", "value"),
        Input("btn-reset", "n_clicks"),
        Input("store-consent", "data"),
    ],
    [
        State("store-trace", "data"),
        State("store-x", "data"),
    ],
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
        traceStore,
        figure,
        xs
    ) {
        const noUpdate = window.dash_clientside.no_update;
        if (!Array.isArray(xs) || xs.length === 0) {
            return [noUpdate, noUpdate, noUpdate];
        }

        const fallback = {a: 1.0, b: 0.0, c: 0.0};
        const fallbackTraceCapacity = 10;
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
                marker: {color: "#D55E00", size: 10, line: {color: "#ffffff", width: 1}},
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
                marker: {color: "#000000", size: 9, symbol: "x", line: {color: "#ffffff", width: 1}},
                hovertemplate: "Zero<br>x=%{x:.2f}<extra></extra>",
                showlegend: false,
            };
        }
        zerosTrace.mode = "markers";
        zerosTrace.showlegend = false;
        if (!zerosTrace.marker) {
            zerosTrace.marker = {color: "#000000", size: 9, symbol: "x", line: {color: "#ffffff", width: 1}};
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
                    line: {color: "#777777", width: 1, dash: "dash"},
                };
            } else {
                vertexTrace.x = [];
                vertexTrace.y = [];
            }
        } else {
            vertexTrace.x = [];
            vertexTrace.y = [];
            if (vertexToggleActive && isDegenerate) {
            vertexNotice = "Vertex/axis hidden when a = 0 (linear case).";
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

        const overlayCapacity =
            traceStore &&
            typeof traceStore.capacity === "number" &&
            traceStore.capacity > 0 &&
            traceStore.capacity <= fallbackTraceCapacity
                ? traceStore.capacity
                : fallbackTraceCapacity;
        const overlayEntries =
            traceStore && Array.isArray(traceStore.entries) ? traceStore.entries : [];
        const overlayStart = 3;
        const ensureOverlayTrace = () => ({
            x: [],
            y: [],
            mode: "lines",
            name: "Trace",
            line: {width: 1.5, color: "rgba(0,114,178,0.35)"},
            hoverinfo: "skip",
            showlegend: false,
        });
        for (var idx = 0; idx < overlayCapacity; idx++) {
            const traceIndex = overlayStart + idx;
            var overlayTrace = newData[traceIndex];
            if (overlayTrace) {
                overlayTrace = Object.assign({}, overlayTrace);
            } else {
                overlayTrace = ensureOverlayTrace();
            }
            const snapshot = overlayEntries[idx];
            if (
                snapshot &&
                Array.isArray(snapshot.y) &&
                Array.isArray(xs) &&
                snapshot.y.length === xs.length
            ) {
                overlayTrace.x = xs;
                overlayTrace.y = snapshot.y;
                const baseOpacity = 0.45;
                const fadeStep = overlayCapacity > 1 ? 0.35 / (overlayCapacity - 1) : 0.35;
                const opacity = Math.max(0.08, baseOpacity - idx * fadeStep);
                overlayTrace.line = Object.assign({}, overlayTrace.line || {}, {
                    color: "rgba(0,114,178," + opacity.toFixed(3) + ")",
                    width: 1.6,
                });
                overlayTrace.hoverinfo = "skip";
                overlayTrace.showlegend = false;
            } else {
                overlayTrace.x = [];
                overlayTrace.y = [];
            }
            newData[traceIndex] = overlayTrace;
        }

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
        Input("store-trace", "data"),
    ],
    [
        State("graph-main", "figure"),
        State("store-x", "data"),
    ],
)


app.clientside_callback(
    """
    (function() {
        const MAX_LEN = %d;
        const SOFT_MIN = %d;
        const defaultDraft = () => ({
            text: "",
            focused_ts: null,
            first_meaningful_edit_ts: null,
            last_edit_ts: null,
            edit_count: 0,
            initial_len: 0,
            paste_flag: false,
        });
        const ensureDraft = (raw) => {
            if (!raw || typeof raw !== "object") {
                return defaultDraft();
            }
            const draft = Object.assign(defaultDraft(), raw);
            draft.text = typeof draft.text === "string" ? draft.text : "";
            draft.focused_ts =
                typeof draft.focused_ts === "number" && isFinite(draft.focused_ts)
                    ? draft.focused_ts
                    : null;
            draft.first_meaningful_edit_ts =
                typeof draft.first_meaningful_edit_ts === "number" && isFinite(draft.first_meaningful_edit_ts)
                    ? draft.first_meaningful_edit_ts
                    : null;
            draft.last_edit_ts =
                typeof draft.last_edit_ts === "number" && isFinite(draft.last_edit_ts)
                    ? draft.last_edit_ts
                    : null;
            draft.edit_count = typeof draft.edit_count === "number" ? draft.edit_count : 0;
            draft.initial_len = typeof draft.initial_len === "number" ? draft.initial_len : 0;
            draft.paste_flag = Boolean(draft.paste_flag);
            return draft;
        };

        return function(
            value,
            clickTs,
            ackData,
            consentData,
            statusData,
            draftState
        ) {
            const ctx = window.dash_clientside && window.dash_clientside.callback_context;
            const triggered = ctx && ctx.triggered && ctx.triggered.length ? ctx.triggered[0].prop_id : "";
            let draft = ensureDraft(draftState);
            let nextValue = typeof value === "string" ? value : "";
            const consentGranted = Boolean(consentData && consentData.granted);
            const status = statusData && typeof statusData === "object" ? statusData : {};
            const pendingSeq = status.pending_seq;
            const submitting = Boolean(status.submitting);
            const ack =
                ackData && typeof ackData === "object" && ackData !== null ? ackData : null;
            const ackSeq = ack && typeof ack.seq === "number" ? ack.seq : null;
            const shouldReset =
                triggered === "store-reflection-ack.data" &&
                ack &&
                ack.status === "ok" &&
                ackSeq !== null &&
                pendingSeq !== null &&
                ackSeq === pendingSeq;

            if (shouldReset) {
                draft = defaultDraft();
                nextValue = "";
            } else {
                if (triggered === "reflect-input.n_clicks_timestamp" && !draft.focused_ts) {
                    if (typeof clickTs === "number" && isFinite(clickTs) && clickTs > 0) {
                        draft.focused_ts = clickTs;
                    } else {
                        draft.focused_ts = Date.now();
                    }
                }
                if (nextValue.length > MAX_LEN) {
                    nextValue = nextValue.slice(0, MAX_LEN);
                }
                const prevText = draft.text || "";
                if (nextValue !== prevText) {
                    const now = Date.now();
                    const prevLen = prevText.length;
                    const newLen = nextValue.length;
                    const delta = Math.abs(newLen - prevLen);
                    draft.text = nextValue;
                    draft.last_edit_ts = now;
                    draft.edit_count = (draft.edit_count || 0) + 1;
                    if (delta >= 20) {
                        draft.paste_flag = true;
                    }
                    const trimmedLen = nextValue.trim().length;
                    if (!draft.first_meaningful_edit_ts) {
                        if (trimmedLen >= 10 || delta >= 3) {
                            draft.first_meaningful_edit_ts = now;
                        }
                    }
                    if (trimmedLen > 0 && (!draft.initial_len || draft.initial_len < 0)) {
                        draft.initial_len = prevLen;
                    } else if (trimmedLen > 0 && draft.initial_len === 0 && prevLen === 0) {
                        draft.initial_len = 0;
                    }
                    if (!draft.focused_ts) {
                        draft.focused_ts = now;
                    }
                }
            }

            const text = draft.text || "";
            const trimmed = text.trim();
            const length = text.length;
            let counterText = `${length} / ${MAX_LEN} characters`;
            if (length >= MAX_LEN) {
                counterText += " (max)";
            }
            let hint = "";
            if (!trimmed.length) {
                hint = "Write at least 20 characters to build a useful log.";
            } else if (trimmed.length < SOFT_MIN) {
                const remaining = SOFT_MIN - trimmed.length;
                hint = `Add ${remaining} more character${remaining === 1 ? "" : "s"} for a stronger reflection.`;
            } else {
                hint = "Great! Ready to submit.";
            }

            let disabled = !consentGranted || trimmed.length === 0 || length > MAX_LEN || submitting;
            let errorText = "";
            if (!consentGranted) {
                errorText = "Please accept consent to enable reflections.";
            } else if (length > MAX_LEN) {
                errorText = "Reflection is too long.";
            } else if (submitting) {
                errorText = "Submitting…";
            }

            return [draft, nextValue, counterText, hint, disabled, errorText];
        };
    })()
    """ % (_REFLECTION_MAX_LEN, _REFLECTION_SOFT_MIN),
    [
        Output("store-reflection-draft", "data"),
        Output("reflect-input", "value"),
        Output("reflect-char-count", "children"),
        Output("reflect-soft-hint", "children"),
        Output("reflect-submit", "disabled"),
        Output("reflect-error-message", "children"),
    ],
    [
        Input("reflect-input", "value"),
        Input("reflect-input", "n_clicks_timestamp"),
        Input("store-reflection-ack", "data"),
        Input("store-consent", "data"),
        Input("store-reflection-status", "data"),
    ],
    State("store-reflection-draft", "data"),
    prevent_initial_call=True,
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
    """
    (function() {
        const formatNumber = (value) => {
            const num = Number(value);
            if (!isFinite(num)) {
                return "?";
            }
            const rounded = Math.round(num * 10) / 10;
            const normalized = Object.is(rounded, -0) ? 0 : rounded;
            const text = normalized.toFixed(1);
            return text.endsWith(".0") ? text.slice(0, -2) : text;
        };
        const isToggleOn = (value) => {
            if (Array.isArray(value)) {
                return value.indexOf("on") !== -1;
            }
            if (value === null || value === undefined) {
                return false;
            }
            return Boolean(value);
        };
        const ensureOrigin = (entry) => {
            if (!window.__traceHistoryOrigin) {
                window.__traceHistoryOrigin =
                    entry && typeof entry.t_client === "number" ? entry.t_client : Date.now();
            }
            return window.__traceHistoryOrigin;
        };
        const formatTime = (entry) => {
            const origin = ensureOrigin(entry);
            const tVal =
                entry && typeof entry.t_client === "number" ? entry.t_client : Date.now();
            const delta = tVal - origin;
            const sign = delta >= 0 ? "+" : "-";
            const seconds = Math.abs(delta) / 1000;
            return sign + seconds.toFixed(1) + "s";
        };
        const formatChange = (entry) => {
            if (!entry || entry.param === "reset") {
                return "baseline (reset)";
            }
            const param = entry.param || "?";
            const oldValue =
                entry.old_value === null || entry.old_value === undefined
                    ? "?"
                    : formatNumber(entry.old_value);
            const newValue =
                entry.new_value === null || entry.new_value === undefined
                    ? "?"
                    : formatNumber(entry.new_value);
            const source = entry.source || "slider";
            return param + " " + oldValue + " → " + newValue + " (" + source + ")";
        };
        return function(traceStore, consentData, toggleValue) {
            const consentGranted = Boolean(consentData && consentData.granted);
            const traceEnabled = isToggleOn(toggleValue);
            if (!traceEnabled) {
                window.__traceHistoryOrigin = null;
                return "Trace mode is off.";
            }
            if (!consentGranted) {
                window.__traceHistoryOrigin = null;
                return "Enable research consent to see trace history.";
            }
            const entries =
                traceStore && Array.isArray(traceStore.entries) ? traceStore.entries : [];
            if (!entries.length) {
                window.__traceHistoryOrigin = null;
                return "Trace history will appear here.";
            }
            ensureOrigin(entries[0]);
            const lines = entries.map((entry) => {
                const timeLabel = formatTime(entry);
                const changeText = formatChange(entry);
                const tuple =
                    "(" +
                    formatNumber(entry.a) +
                    ", " +
                    formatNumber(entry.b) +
                    ", " +
                    formatNumber(entry.c) +
                    ")";
                return timeLabel + " - " + changeText + " | " + tuple;
            });
            return lines.join("\\n");
        };
    })()
    """,
    Output("trace-history", "children"),
    [
        Input("store-trace", "data"),
        Input("store-consent", "data"),
        Input("toggle-trace", "value"),
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


app.clientside_callback(
    """
    (function() {
        const baseStyle = {
            marginLeft: "12px",
            fontSize: "0.9rem",
            color: "#2e8b57",
            opacity: 0,
            transition: "opacity 0.25s ease-in-out",
        };
        const ensureStatus = (raw) => {
            if (!raw || typeof raw !== "object") {
                return {};
            }
            return raw;
        };
        return function(statusData, intervalTicks) {
            void intervalTicks;
            const status = ensureStatus(statusData);
            const until =
                status && typeof status.saved_until === "number" && isFinite(status.saved_until)
                    ? status.saved_until
                    : null;
            const now = Date.now();
            const visible = until !== null && now < until;
            const indicatorStyle = Object.assign({}, baseStyle, {
                opacity: visible ? 1 : 0,
            });
            const disableInterval = visible ? false : true;
            return [indicatorStyle, disableInterval];
        };
    })()
    """,
    [
        Output("reflect-saved-indicator", "style"),
        Output("interval-reflect-saved", "disabled"),
    ],
    [
        Input("store-reflection-status", "data"),
        Input("interval-reflect-saved", "n_intervals"),
    ],
)


@app.callback(
    [
        Output("store-reflection-ack", "data"),
        Output("store-log-sink", "data", allow_duplicate=True),
    ],
    Input("store-reflection-commit", "data"),
    [
        State("store-session", "data"),
        State("store-consent", "data"),
        State("store-ui", "data"),
        State("store-log-sink", "data"),
    ],
    prevent_initial_call=True,
)
def _log_reflection_submit(commit_data, session_data, consent_data, ui_store_data, log_store_data):
    if not isinstance(commit_data, dict):
        return dash.no_update, dash.no_update
    seq = commit_data.get("seq")
    try:
        seq_int = int(seq)
    except (TypeError, ValueError):
        return dash.no_update, dash.no_update

    session_id = _get_session_id(session_data)
    safe_id = _safe_session_id(session_id)
    ack_payload: Dict[str, Any] = {"seq": seq_int, "status": "noop"}
    if not _is_logging_allowed(consent_data):
        ack_payload["status"] = "blocked_no_consent"
        return ack_payload, dash.no_update

    last_seq = _LAST_REFLECTION_SEQ.get(safe_id, 0)
    if seq_int <= last_seq:
        ack_payload["status"] = "duplicate"
        return ack_payload, dash.no_update

    reflection_text = (commit_data.get("reflection_text") or "").strip()
    if not reflection_text:
        ack_payload["status"] = "empty"
        return ack_payload, dash.no_update

    chars = commit_data.get("chars")
    try:
        chars_int = int(chars)
    except (TypeError, ValueError):
        chars_int = len(reflection_text)
    words = commit_data.get("words")
    try:
        words_int = int(words)
    except (TypeError, ValueError):
        words_int = len(reflection_text.split())

    def _int_or_none(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed

    extras = {
        "reflection_text": reflection_text,
        "chars": chars_int,
        "words": words_int,
        "draft_total_ms": _int_or_none(commit_data.get("draft_total_ms")),
        "idle_to_submit_ms": _int_or_none(commit_data.get("idle_to_submit_ms")),
        "edit_count": _int_or_none(commit_data.get("edit_count")),
        "char_delta": _int_or_none(commit_data.get("char_delta")),
        "paste_flag": bool(commit_data.get("paste_flag")),
        "accidental_focus_flag": bool(commit_data.get("accidental_focus_flag")),
        "t_client": _int_or_none(commit_data.get("t_client")),
    }
    record = _base_log_record(
        session_id,
        event="reflection_submit",
        param_name=None,
        old_value=None,
        new_value=None,
        source="input",
        uirevision=_resolve_uirevision_value(ui_store_data),
        viewport=None,
        extras=extras,
    )
    _write_log_record(session_id, record)
    _LAST_REFLECTION_SEQ[safe_id] = seq_int
    ack_payload["status"] = "ok"
    summary = _format_preview_message(record)
    return ack_payload, _append_preview_log(log_store_data, summary)


@app.callback(
    Output("store-log-sink", "data", allow_duplicate=True),
    Input("store-source", "data"),
    [
        State("slider-a", "value"),
        State("slider-b", "value"),
        State("slider-c", "value"),
        State("store-session", "data"),
        State("store-consent", "data"),
        State("store-ui", "data"),
        State("store-log-sink", "data"),
    ],
    prevent_initial_call=True,
)
def _log_slider_activity(source_store, a_value, b_value, c_value, session_data, consent_data, ui_store_data, log_store_data):
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
        Input("toggle-trace", "value"),
    ],
    [
        State("store-session", "data"),
        State("store-consent", "data"),
        State("store-ui", "data"),
        State("store-log-sink", "data"),
    ],
    prevent_initial_call=True,
)
def _log_overlay_toggle(vertex_value, zeros_value, trace_value, session_data, consent_data, ui_store_data, log_store_data):
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
    elif trigger_id == "toggle-trace":
        overlay_key = "trace"
        new_enabled = _is_toggle_enabled(trace_value)
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
