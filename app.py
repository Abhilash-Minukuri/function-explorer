import streamlit as st

st.set_page_config(page_title="Function Explorer", layout="wide")

st.title("Function Explorer — Quadratic MVP")
st.caption("Scaffold placeholder — controls/graph/representations panes")

left, right = st.columns([1, 2])
with left:
    st.header("Controls")
    st.write("Sliders go here (a, b, c).")
with right:
    st.header("Graph")
    st.write("Plotly chart placeholder.")
with st.container():
    st.header("Representations")
    st.write("Equation (LaTeX) + verbal feedback + reflection box.")
