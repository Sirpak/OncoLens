from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import streamlit as st
import yaml
from matplotlib import pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_image_table
from src.inference import generate_gradcam, load_checkpoint, predict_image


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@st.cache_resource
def cached_model(model_path: str):
    return load_checkpoint(model_path)


def load_metrics(path: str):
    if not path or not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


st.set_page_config(page_title="OncoLens Dashboard", layout="wide")
cfg = load_yaml(ROOT / "config" / "config.yaml")
st.title("OncoLens Histopathology Dashboard")

model_path = st.sidebar.text_input("Model checkpoint", "")
metrics_path = st.sidebar.text_input("Metrics JSON", cfg["dashboard"].get("metrics_path", ""))
sample_dir = st.sidebar.text_input("External sample image directory", cfg["dashboard"].get("sample_image_dir", ""))

bundle = None
if model_path:
    try:
        bundle = cached_model(model_path)
        st.sidebar.success("Checkpoint loaded")
    except Exception as exc:
        st.sidebar.error(f"Unable to load checkpoint: {exc}")

metrics = load_metrics(metrics_path)
if metrics:
    st.subheader("Training Metrics")
    history = pd.DataFrame(metrics["history"])
    st.line_chart(history)
    st.subheader("Class Distribution")
    st.bar_chart(pd.Series(metrics["class_distribution"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Validation Confusion Matrix")
    st.pyplot(fig)

uploaded = st.file_uploader("Upload a pathology image", type=["png", "jpg", "jpeg"])
image = Image.open(uploaded).convert("RGB") if uploaded else None

if image is None and sample_dir:
    try:
        sample_table = load_image_table(sample_dir)
        subset = sample_table.groupby("label", group_keys=False).head(2).reset_index(drop=True)
        st.subheader("Example Histopathology Gallery")
        cols = st.columns(3)
        for idx, row in subset.head(6).iterrows():
            cols[idx % 3].image(row["image_path"], caption=row["label"], use_container_width=True)
        options = {f"{row['label']} :: {Path(row['image_path']).name}": row["image_path"] for _, row in subset.head(24).iterrows()}
        selected = st.selectbox("Or choose a sample image", [""] + list(options.keys()))
        if selected:
            image = Image.open(options[selected]).convert("RGB")
    except Exception as exc:
        st.info(f"Sample gallery unavailable: {exc}")

if image and bundle:
    prediction = predict_image(bundle, image)
    gradcam = generate_gradcam(bundle, image)
    left, right = st.columns(2)
    left.image(image, caption="Input image", use_container_width=True)
    right.image(gradcam, caption="Grad-CAM overlay", use_container_width=True)
    st.metric("Predicted class", prediction["predicted_class"], f"confidence {prediction['confidence']:.2%}")
    st.subheader("Class Probabilities")
    st.bar_chart(pd.Series(prediction["probabilities"]))
elif image and not bundle:
    st.warning("Load a trained model checkpoint to run predictions.")
else:
    st.caption("Upload an image or select one from the external sample gallery.")