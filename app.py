import streamlit as st
import plotly.graph_objects as go

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
    if sync_slider_state:
        st.session_state[f"param_{name}_slider"] = val
    return val

def _sync_param_from_widgets(name: str):
    cfg = _PARAM_CFG[name]
    key = f"param_{name}"
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    canonical = st.session_state[key]
    slider_val = st.session_state.get(slider_key, canonical)
    slider_val = _sanitize_value(slider_val, min_v=cfg["min"], max_v=cfg["max"], step=cfg["step"])
    if slider_val != canonical:
        _apply_param_value(name, slider_val, sync_slider_state=False)
        canonical = st.session_state[key]

    input_val = st.session_state.get(input_key, canonical)
    input_val = _sanitize_value(input_val, min_v=cfg["min"], max_v=cfg["max"], step=cfg["step"])
    if input_val != canonical:
        _apply_param_value(name, input_val, sync_slider_state=True)

# Initialize authoritative values and widget states
for _n, _cfg in _PARAM_CFG.items():
    key = f"param_{_n}"
    slider_key = f"{key}_slider"
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

    st.session_state.setdefault(slider_key, st.session_state[key])
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

    # Quadratic parameters (UI-only, synced slider + numeric input)
    st.subheader("Quadratic parameters")
    if st.button("Reset view & params", use_container_width=True):
        for _n, _cfg in _PARAM_CFG.items():
            _apply_param_value(_n, _cfg["default"], sync_slider_state=True)
        st.session_state["graph_reset_view"] = True

    is_quadratic_active = st.session_state.get("function_type", "quadratic") == "quadratic"

    # a
    cfg = _PARAM_CFG["a"]
    _sync_param_from_widgets("a")
    st.slider(
        "a",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        key="param_a_slider",
        disabled=not is_quadratic_active,
    )
    st.number_input(
        "a (exact)",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        format="%.1f",
        key="param_a_input",
        disabled=not is_quadratic_active,
    )

    # b
    cfg = _PARAM_CFG["b"]
    _sync_param_from_widgets("b")
    st.slider(
        "b",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        key="param_b_slider",
        disabled=not is_quadratic_active,
    )
    st.number_input(
        "b (exact)",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        format="%.1f",
        key="param_b_input",
        disabled=not is_quadratic_active,
    )

    # c
    cfg = _PARAM_CFG["c"]
    _sync_param_from_widgets("c")
    st.slider(
        "c",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        key="param_c_slider",
        disabled=not is_quadratic_active,
    )
    st.number_input(
        "c (exact)",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        format="%.1f",
        key="param_c_input",
        disabled=not is_quadratic_active,
    )

with right_col:
    st.header("Graph")
    # Dynamic quadratic plot based on current parameters
    a = st.session_state["param_a"]
    b = st.session_state["param_b"]
    c = st.session_state["param_c"]
    x_min, x_max = -10.0, 10.0
    num_samples = 501
    step = (x_max - x_min) / (num_samples - 1)
    xs = [x_min + i * step for i in range(num_samples)]
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
        fig.update_xaxes(range=[x_min, x_max])
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
