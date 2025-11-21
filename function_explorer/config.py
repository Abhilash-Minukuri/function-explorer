from __future__ import annotations

from pathlib import Path

# Paths and filenames
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "function_explorer" / "data"

# Quadratic sampling grid
X_MIN = -10.0
X_MAX = 10.0
NUM_SAMPLES = 401

# Parameter defaults and bounds
DEFAULT_PARAMS = {"a": 1.0, "b": 0.0, "c": 0.0}
PARAM_BOUNDS = {
    "b": {"min": -10.0, "max": 10.0, "step": 0.1},
    "a": {"min": -5.0, "max": 5.0, "step": 0.1},
    "c": {"min": -10.0, "max": 10.0, "step": 0.1},
}

# Precision / guard rails
EPS_ZERO = 1e-6

# UI, mode, schema
UI_BASE_TOKEN = "quadratic-"
DEFAULT_UI_NONCE = "0"
DEFAULT_CONSENT_STATE = {"granted": False, "declined": False, "timestamp_utc": None}
SCHEMA_VERSION = 1
FUNCTION_TYPE = "quadratic"
APP_MODE = "dash"
DEFAULT_INTERACTION_PHASE = "change"

# Logging and tracing
TRACE_HISTORY_CAPACITY = 10
TRACE_STORE_DEFAULT = {"entries": [], "capacity": TRACE_HISTORY_CAPACITY}
LOG_RATE_LIMIT_SECONDS = 0.1

# CSV column order (unchanged)
SCHEMA_COLUMNS = [
    "schema_version",
    "session_id",
    "t_client_ms",
    "t_server_iso",
    "seq",
    "event",
    "function_type",
    "param_name",
    "old_value",
    "new_value",
    "source",
    "a",
    "b",
    "c",
    "elapsed_time_ms",
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
    "consent_status",
    "export_type",
    "overlay",
    "enabled",
]

# Plot palette and styles (Okabe–Ito)
FIGURE_COLORS = {
    "curve": "#0072B2",
    "vertex": "#D55E00",
    "zeros": "#000000",
    "trace": "rgba(0,114,178,0.35)",
}
CURVE_LINE_STYLE = {"color": FIGURE_COLORS["curve"], "width": 3}
VERTEX_MARKER_STYLE = {
    "color": FIGURE_COLORS["vertex"],
    "size": 10,
    "symbol": "circle",
    "line": {"color": "#ffffff", "width": 1},
}
ZERO_MARKER_STYLE = {
    "color": FIGURE_COLORS["zeros"],
    "size": 9,
    "symbol": "x",
    "line": {"color": "#ffffff", "width": 1},
}
TRACE_LINE_STYLE = {"color": FIGURE_COLORS["trace"], "width": 1.5}
AXIS_LINE_STYLE = {"zerolinecolor": "#777777"}

# External assets
MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
