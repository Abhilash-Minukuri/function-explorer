import streamlit as st
import plotly.graph_objects as go
import streamlit_shadcn_ui as ui

# Wide layout and clear page title for this step
st.set_page_config(page_title="Quadratic MVP - Layout Skeleton", layout="wide")

st.title("Quadratic MVP - Layout Skeleton")
st.caption("Layout-only: controls (left), graph placeholder (right), representations (bottom).")

# Session state for function type (persisted)
if "function_type" not in st.session_state:
    st.session_state["function_type"] = "quadratic"
if "function_type_choice" not in st.session_state:
    st.session_state["function_type_choice"] = "Quadratic"
if "function_type_notice" not in st.session_state:
    st.session_state["function_type_notice"] = False
if "graph_reset_view" not in st.session_state:
    st.session_state["graph_reset_view"] = False

def _on_function_type_change():
    choice = st.session_state.get("function_type_choice", "Quadratic")
    if choice == "Quadratic":
        st.session_state["function_type"] = "quadratic"
        st.session_state["function_type_notice"] = False
    else:
        # Revert selection and inform the user (one-time info message)
        st.session_state["function_type_choice"] = "Quadratic"
        st.session_state["function_type_notice"] = True

# Quadratic parameter configuration and helpers
_PARAM_CFG = {
    "a": {"min": -5.0, "max": 5.0, "step": 0.1, "default": 1.0},
    "b": {"min": -10.0, "max": 10.0, "step": 0.1, "default": 0.0},
    "c": {"min": -10.0, "max": 10.0, "step": 0.1, "default": 0.0},
}

_X_MIN, _X_MAX = -10.0, 10.0
_NUM_SAMPLES = 401  # reuse a lighter grid to keep UI snappy during fast drags
_X_STEP = (_X_MAX - _X_MIN) / (_NUM_SAMPLES - 1)
_X_GRID = [_X_MIN + i * _X_STEP for i in range(_NUM_SAMPLES)]

def _sanitize_value(value: float, *, min_v: float, max_v: float, step: float) -> float:
    try:
        v = float(value)
    except Exception:
        # Restore to within range using min as safe fallback
        v = min_v
    # Clamp first
    if v < min_v:
        v = min_v
    if v > max_v:
        v = max_v
    # Normalize precision to match step (1 decimal place)
    v = float(f"{round(v / step) * step:.1f}")
    return v

def _apply_param_value(name: str, value: float, *, sync_slider_state: bool):
    cfg = _PARAM_CFG[name]
    val = _sanitize_value(value, min_v=cfg["min"], max_v=cfg["max"], step=cfg["step"])
    st.session_state[f"param_{name}"] = val
    st.session_state[f"param_{name}_input"] = val
    st.session_state[f"param_{name}_slider_last"] = val
    if sync_slider_state:
        st.session_state[f"param_{name}_slider_force_sync"] = True
        slider_state_key = f"param_{name}_slider"
        if slider_state_key in st.session_state:
            del st.session_state[slider_state_key]

    return val

def _handle_input_change(name: str):
    value = st.session_state.get(f"param_{name}_input")
    if value is not None:
        _apply_param_value(name, value, sync_slider_state=True)

def _reset_params_to_defaults():
    for _n, _cfg in _PARAM_CFG.items():
        _apply_param_value(_n, _cfg["default"], sync_slider_state=True)
    st.session_state["graph_reset_view"] = True
    rerun_fn = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
    if rerun_fn is not None:
        rerun_fn()

def _render_param_controls(name: str, *, slider_label: str, input_label: str, disabled: bool):
    cfg = _PARAM_CFG[name]
    last_key = f"param_{name}_slider_last"
    force_key = f"param_{name}_slider_force_sync"
    force_sync = st.session_state.get(force_key, False)
    slider_default = [st.session_state[f"param_{name}"]] if force_sync else None
    slider_value = ui.slider(
        label=slider_label,
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        default_value=slider_default,
        key=f"param_{name}_slider",
    )
    current_value = st.session_state[f"param_{name}"]
    if not disabled and not force_sync:
        raw_value = None
        if isinstance(slider_value, (list, tuple)):
            if slider_value:
                raw_value = slider_value[0]
        elif isinstance(slider_value, (int, float)):
            raw_value = slider_value
        elif isinstance(slider_value, str):
            try:
                raw_value = float(slider_value)
            except ValueError:
                raw_value = None

        if raw_value is not None:
            try:
                raw_value = float(raw_value)
            except (TypeError, ValueError):
                raw_value = None

        if raw_value is None:
            raw_value = st.session_state.get(last_key, current_value)
        else:
            st.session_state[last_key] = raw_value

        if raw_value is not None:
            rounded_value = round(raw_value, 2)
            if rounded_value != current_value:
                current_value = _apply_param_value(name, rounded_value, sync_slider_state=False)
    else:
        st.session_state[last_key] = current_value

    if force_sync:
        st.session_state[force_key] = False

    input_value = st.number_input(
        input_label,
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        format="%.1f",
        key=f"param_{name}_input",
        disabled=disabled,
        on_change=_handle_input_change,
        args=(name,),
    )
    sanitized_input = _sanitize_value(
        input_value,
        min_v=cfg["min"],
        max_v=cfg["max"],
        step=cfg["step"],
    )
    if sanitized_input != st.session_state[f"param_{name}"]:
        current_value = _apply_param_value(name, sanitized_input, sync_slider_state=True)

    return current_value

# Initialize authoritative values and widget states
for _n, _cfg in _PARAM_CFG.items():
    key = f"param_{_n}"
    slider_last_key = f"{key}_slider_last"
    slider_force_key = f"{key}_slider_force_sync"
    input_key = f"{key}_input"

    if key not in st.session_state:
        st.session_state[key] = _cfg["default"]
    else:
        st.session_state[key] = _sanitize_value(
            st.session_state[key],
            min_v=_cfg["min"],
            max_v=_cfg["max"],
            step=_cfg["step"],
        )

    st.session_state.setdefault(slider_last_key, st.session_state[key])
    st.session_state.setdefault(slider_force_key, True)
    st.session_state.setdefault(input_key, st.session_state[key])

# Two-column layout: 1:2 ratio
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.header("Controls")
    # Function type selector (Quadratic active; others coming soon)
    st.radio(
        "Function type",
        options=[
            "Quadratic",
            "Linear (coming soon)",
            "Exponential (coming soon)",
        ],
        key="function_type_choice",
        on_change=_on_function_type_change,
    )
    if st.session_state.get("function_type_notice"):
        st.info("Linear and Exponential are coming in Phase 2.")
        # Reset the notice flag so it shows briefly once
        st.session_state["function_type_notice"] = False
    st.caption("Tip: For clearer patterns, drag the sliders slowly, nudge with arrow keys, or type exact values.")

    # Quadratic parameters (UI-only, synced slider + numeric input)
    st.subheader("Quadratic parameters")
    if st.button("Reset view & params", use_container_width=True):
        _reset_params_to_defaults()

    is_quadratic_active = st.session_state.get("function_type", "quadratic") == "quadratic"

    # a
    _render_param_controls(
        "a",
        slider_label="a",
        input_label="a (exact)",
        disabled=not is_quadratic_active,
    )

    # b
    _render_param_controls(
        "b",
        slider_label="b",
        input_label="b (exact)",
        disabled=not is_quadratic_active,
    )

    # c
    _render_param_controls(
        "c",
        slider_label="c",
        input_label="c (exact)",
        disabled=not is_quadratic_active,
    )

with right_col:
    st.header("Graph")
    st.caption("Hint: Zoom or pan to inspect details; double-click the plot to reset the view.")
    # Dynamic quadratic plot based on current parameters
    a = st.session_state["param_a"]
    b = st.session_state["param_b"]
    c = st.session_state["param_c"]
    xs = _X_GRID
    ys = [a * (x ** 2) + b * x + c for x in xs]

    fig = go.Figure(
        data=go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="y = x^2",
        )
    )
    fig.update_layout(
        height=560,
        margin=dict(l=36, r=16, t=24, b=32),
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
        uirevision="quadratic",
        transition=dict(duration=250, easing="linear"),
    )
    if st.session_state.pop("graph_reset_view", False):
        y_min = min(ys)
        y_max = max(ys)
        span = y_max - y_min
        padding = max(1.0, span * 0.05) if span > 0 else 1.0
        fig.update_xaxes(range=[_X_MIN, _X_MAX])
        fig.update_yaxes(range=[y_min - padding, y_max + padding])

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

# Clear separation before representations
st.divider()

st.header("Representations")
st.write("- Equation (LaTeX) - placeholder")
st.write("- Verbal description - placeholder")
st.write("- Reflection - placeholder")
