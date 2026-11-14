"""Small demo UI.   streamlit run stage4/demo/streamlit_app.py"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from stage4.inference import Predictor

MODEL_DIR = os.environ.get("MINIJEV_DIR", "artifacts/final")
BACKEND = os.environ.get("MINIJEV_BACKEND", "onnx_int8")

st.set_page_config(page_title="Mini Jev: ASRS triage", layout="wide")
st.title("Mini Jev: aviation safety report triage")
st.caption("A distilled BERT-mini chooses one of the dataset's 18 report categories. "
           "Learning project: not for real safety decisions.")


@st.cache_resource
def load(backend: str):
    return Predictor(MODEL_DIR, backend)


backend = st.sidebar.selectbox("Backend", ["onnx_int8", "onnx", "torch"], index=["onnx_int8", "onnx", "torch"].index(BACKEND))
predictor = load(backend)
text = st.text_area("Report narrative", height=220, placeholder="Paste an ASRS-style narrative here...")

if st.button("Analyse", type="primary") and text.strip():
    t0 = time.perf_counter()
    res = predictor.predict([text])[0]
    ms = (time.perf_counter() - t0) * 1000
    cols = st.columns(3)
    cols[0].metric("Needs review", "YES" if res["needs_review"] else "no")
    cols[1].metric("Latency", f"{ms:.0f} ms")
    cols[2].metric("Reasons", ", ".join(res["review_reasons"]) or "none")
    st.subheader(f"{res['label']}  ·  {res['confidence']:.1%} confidence")
    chart_rows = [{"class": name, "probability": probability}
                  for name, probability in res["probabilities"].items()]
    st.bar_chart(chart_rows, x="class", y="probability")
