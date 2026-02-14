"""
SportsAnalytics-CV Streamlit Demo

Author: Malav Patel
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SportsAnalytics-CV", page_icon="SV", layout="wide")

st.title("SportsAnalytics-CV")
st.markdown("**Real-Time Sports Analytics with Computer Vision**")

st.sidebar.header("Configuration")
model_option = st.sidebar.selectbox(
    "Select Model", ["yolov8n.pt", "yolov8m.pt", "yolov8x.pt", "custom (best.pt)"]
)

device = st.sidebar.radio("Device", ["cuda", "cpu"])
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

st.header("Upload Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("Analyze Video"):
        st.info("Analysis started... This may take a few minutes.")
        # Add analysis code here
        st.success("Analysis complete!")

st.markdown("---")
st.markdown("**Author:** Malav Patel | [GitHub](https://github.com/mlvpatel)")
