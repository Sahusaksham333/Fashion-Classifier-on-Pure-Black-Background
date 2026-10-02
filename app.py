"""Fashion-MNIST classifier – Streamlit interface.

Run:  streamlit run app.py
Needs fashion_cnn.keras (create it first with:  python train_model.py)
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Fashion Classifier", page_icon="👕", layout="wide")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MODEL_PATH = "fashion_cnn.keras"
TEST_CSV = next((p for p in ("input/fashion-mnist_test.csv", "fashion-mnist_test.csv")
                 if os.path.exists(p)), None)


# ----------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import keras
    return keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner="Loading test set…")
def load_test_data():
    arr = pd.read_csv(TEST_CSV).to_numpy("float32")
    return arr[:, 1:] / 255.0, arr[:, 0].astype("int32")


@st.cache_data(show_spinner="Evaluating on the 10,000 test images…")
def evaluate_per_class():
    X, y = load_test_data()
    pred = np.argmax(load_model().predict(X.reshape(-1, 28, 28, 1), verbose=0), axis=1)
    acc = [(pred[y == c] == c).mean() for c in range(10)]
    cm = pd.crosstab(pd.Series(y, name="True"), pd.Series(pred, name="Predicted"))
    cm = cm.reindex(index=range(10), columns=range(10), fill_value=0)
    cm.index, cm.columns = CLASS_NAMES, CLASS_NAMES
    return pd.DataFrame({"Class": CLASS_NAMES, "Accuracy": acc}).set_index("Class"), cm, (pred == y).mean()


# ----------------------------------------------------------- preprocessing
def preprocess(img: Image.Image, auto_invert: bool, crop: bool) -> np.ndarray:
    """Turn an arbitrary photo into a 28x28 float array shaped like Fashion-MNIST
    (light garment on a dark background)."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    gray = np.asarray(img.convert("L"), dtype="float32")

    if auto_invert:
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        if border.mean() > 127:          # bright background -> flip to dark background
            gray = 255.0 - gray

    if crop:
        mask = gray > max(30.0, gray.max() * 0.2)
        if mask.any():
            ys, xs = np.where(mask)
            gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        side = int(max(gray.shape) * 1.15)   # square canvas with a small margin
        canvas = np.zeros((side, side), dtype="float32")
        y0, x0 = (side - gray.shape[0]) // 2, (side - gray.shape[1]) // 2
        canvas[y0:y0 + gray.shape[0], x0:x0 + gray.shape[1]] = gray
        gray = canvas

    small = Image.fromarray(gray.astype("uint8")).resize((28, 28), Image.LANCZOS)
    return np.asarray(small, dtype="float32") / 255.0


def predict(arr28: np.ndarray) -> np.ndarray:
    return load_model().predict(arr28.reshape(1, 28, 28, 1), verbose=0)[0]


def show_prediction(probs: np.ndarray):
    top = int(np.argmax(probs))
    conf = float(probs[top])
    st.metric("Prediction", CLASS_NAMES[top], f"{conf:.1%} confidence", delta_color="off")
    if conf < 0.60:
        st.warning("Low confidence – the image may be ambiguous or not a garment.")
    chart = pd.DataFrame({"Probability": probs}, index=CLASS_NAMES)
    st.bar_chart(chart, horizontal=True, height=330)


# ------------------------------------------------------------------- guard
if not os.path.exists(MODEL_PATH):
    st.error(f"`{MODEL_PATH}` not found. Run `python train_model.py` first to train and save the model.")
    st.stop()

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("👕 Fashion Classifier")
    st.caption("CNN trained on Fashion-MNIST (28×28 grayscale, 10 classes).")
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            st.metric("Test accuracy", f"{json.load(f)['test_accuracy']:.2%}")
    st.subheader("Upload preprocessing")
    auto_invert = st.checkbox("Auto-invert bright backgrounds", value=True,
                              help="The dataset has light garments on a dark background.")
    crop = st.checkbox("Crop to item & centre", value=True)
    st.divider()
    st.caption("Classes: " + ", ".join(CLASS_NAMES))

tab_upload, tab_test, tab_perf = st.tabs(["📤 Upload an image", "🎲 Test-set explorer", "📊 Model performance"])

# ------------------------------------------------------------------ upload
with tab_upload:
    st.write("Upload a clear photo of a single clothing item, ideally on a plain background.")
    source = st.radio("Source", ["Upload file", "Use camera"], horizontal=True, label_visibility="collapsed")
    file = (st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"],
                             label_visibility="collapsed")
            if source == "Upload file" else st.camera_input("Take a photo", label_visibility="collapsed"))
    if file is not None:
        try:
            original = Image.open(file)
            original.load()
        except Exception:
            st.error("That file couldn't be read as an image.")
        else:
            arr = preprocess(original, auto_invert, crop)
            c1, c2, c3 = st.columns([1, 1, 1.4])
            c1.image(original, caption="Original", width="stretch")
            c2.image(arr, caption="What the model sees (28×28)", clamp=True, width=224)
            with c3:
                show_prediction(predict(arr))

# -------------------------------------------------------------- test set
with tab_test:
    if TEST_CSV is None:
        st.info("Place `fashion-mnist_test.csv` in `input/` (or next to app.py) to use this tab.")
    else:
        X, y = load_test_data()
        if "idx" not in st.session_state:
            st.session_state.idx = int(np.random.randint(len(X)))

        def pick_random():
            st.session_state.idx = int(np.random.randint(len(X)))

        left, right = st.columns([1, 3])
        left.button("🎲 Random image", on_click=pick_random, width="stretch")
        right.number_input("…or choose an index", 0, len(X) - 1, step=1, key="idx")
        i = int(st.session_state.idx)
        probs = predict(X[i])
        pred = int(np.argmax(probs))
        c1, c2 = st.columns([1, 1.6])
        c1.image(X[i].reshape(28, 28), caption=f"Test image #{i}", clamp=True, width=224)
        with c2:
            show_prediction(probs)
            if pred == y[i]:
                st.success(f"Correct – true label: **{CLASS_NAMES[y[i]]}**")
            else:
                st.error(f"Wrong – true label: **{CLASS_NAMES[y[i]]}**")

# ----------------------------------------------------------- performance
with tab_perf:
    if TEST_CSV is None:
        st.info("The test CSV is required for this tab.")
    else:
        per_class, cm, overall = evaluate_per_class()
        st.metric("Overall test accuracy", f"{overall:.2%}")
        a, b = st.columns(2)
        a.subheader("Accuracy by class")
        a.bar_chart(per_class, horizontal=True)
        b.subheader("Confusion matrix")
        b.dataframe(cm.style.background_gradient(cmap="Blues"), width="stretch")
        st.caption("Shirt, T-shirt/top, Pullover and Coat are the classes most often confused with each other.")
