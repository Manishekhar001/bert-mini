"""Small demo UI.   streamlit run stage4/demo/streamlit_app.py"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from stage4.inference import Predictor

HF_REPO = os.environ.get("BERTMINI_HF_REPO")        # e.g. yourname/mini-bert-asrs (set on Render)
LOCAL_DIR = os.environ.get("BERTMINI_DIR", "artifacts/final_pretrained")
BACKEND = os.environ.get("BERTMINI_BACKEND", "onnx_int8")

st.set_page_config(page_title="bert-mini: ASRS triage", layout="wide")
st.title("bert-mini: aviation safety report triage")
st.caption("A distilled BERT-mini chooses one of the dataset's 18 report categories. "
           "Not for real safety decisions.")


@st.cache_resource(show_spinner="Downloading model...")
def model_dir() -> str:
    """Local package folder, or the onnx/ folder of the Hugging Face repo when BERTMINI_HF_REPO is set."""
    if not HF_REPO:
        return LOCAL_DIR
    from huggingface_hub import snapshot_download

    root = snapshot_download(HF_REPO, allow_patterns=["onnx/*"], token=os.environ.get("HF_TOKEN") or None)
    return str(Path(root) / "onnx")


def available_backends(directory: str) -> list[str]:
    files = {"onnx_int8": "model.int8.onnx", "onnx": "model.onnx", "torch": "model.pt"}
    return [name for name, file in files.items() if (Path(directory) / file).exists()]


@st.cache_resource
def load(directory: str, backend: str):
    return Predictor(directory, backend)


directory = model_dir()
options = available_backends(directory)
if not options:
    st.error(f"No model files found in {directory}.")
    st.stop()
backend = st.sidebar.selectbox("Backend", options, index=options.index(BACKEND) if BACKEND in options else 0)
predictor = load(directory, backend)
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
