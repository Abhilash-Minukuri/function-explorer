import streamlit as st

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

def _slider_changed(name: str):
    cfg = _PARAM_CFG[name]
    raw = st.session_state.get(f"param_{name}_slider", st.session_state.get(f"param_{name}", cfg["default"]))
    val = _sanitize_value(raw, min_v=cfg["min"], max_v=cfg["max"], step=cfg["step"])
    st.session_state[f"param_{name}"] = val
    st.session_state[f"param_{name}_slider"] = val
    st.session_state[f"param_{name}_input"] = val

def _input_changed(name: str):
    cfg = _PARAM_CFG[name]
    raw = st.session_state.get(f"param_{name}_input", st.session_state.get(f"param_{name}", cfg["default"]))
    val = _sanitize_value(raw, min_v=cfg["min"], max_v=cfg["max"], step=cfg["step"])
    st.session_state[f"param_{name}"] = val
    st.session_state[f"param_{name}_slider"] = val
    st.session_state[f"param_{name}_input"] = val

# Initialize authoritative values and widget states
for _n, _cfg in _PARAM_CFG.items():
    key = f"param_{_n}"
    if key not in st.session_state:
        st.session_state[key] = _cfg["default"]
    # Seed widget states to the authoritative value if missing
    st.session_state.setdefault(f"param_{_n}_slider", st.session_state[key])
    st.session_state.setdefault(f"param_{_n}_input", st.session_state[key])

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

    is_quadratic_active = st.session_state.get("function_type", "quadratic") == "quadratic"

    # a
    cfg = _PARAM_CFG["a"]
    st.slider(
        "a",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        key="param_a_slider",
        on_change=_slider_changed,
        args=("a",),
        disabled=not is_quadratic_active,
    )
    st.number_input(
        "a (exact)",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        format="%.1f",
        key="param_a_input",
        on_change=_input_changed,
        args=("a",),
        disabled=not is_quadratic_active,
    )

    # b
    cfg = _PARAM_CFG["b"]
    st.slider(
        "b",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        key="param_b_slider",
        on_change=_slider_changed,
        args=("b",),
        disabled=not is_quadratic_active,
    )
    st.number_input(
        "b (exact)",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        format="%.1f",
        key="param_b_input",
        on_change=_input_changed,
        args=("b",),
        disabled=not is_quadratic_active,
    )

    # c
    cfg = _PARAM_CFG["c"]
    st.slider(
        "c",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        key="param_c_slider",
        on_change=_slider_changed,
        args=("c",),
        disabled=not is_quadratic_active,
    )
    st.number_input(
        "c (exact)",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        format="%.1f",
        key="param_c_input",
        on_change=_input_changed,
        args=("c",),
        disabled=not is_quadratic_active,
    )

with right_col:
    st.header("Graph")
    # Fixed-height placeholder area (prevents reflow when the chart is added later)
    st.markdown(
        """
        <div style="
            height: 560px;
            width: 100%;
            border: 1px dashed #AAAAAA;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0,0,0,0.03);
            color: #666666;">
            Future Plotly chart placeholder (560px high)
        </div>
        """,
        unsafe_allow_html=True,
    )

# Clear separation before representations
st.divider()

st.header("Representations")
st.write("- Equation (LaTeX) - placeholder")
st.write("- Verbal description - placeholder")
st.write("- Reflection - placeholder")
