"""Streamlit UI components (stub)."""

import streamlit as st

def quadratic_controls(defaults=(1.0, 0.0, 0.0)):
    a0, b0, c0 = defaults
    a = st.slider("a", -5.0, 5.0, a0, 0.1)
    b = st.slider("b", -10.0, 10.0, b0, 0.1)
    c = st.slider("c", -10.0, 10.0, c0, 0.1)
    return a, b, c
