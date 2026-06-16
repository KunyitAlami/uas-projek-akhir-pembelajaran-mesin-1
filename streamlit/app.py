import streamlit as st
import numpy as np
from PIL import Image
import time

st.set_page_config(
    page_title="Klasifikasi Stopkontak Listrik",
    page_icon="🔌",
    layout="centered",
)

st.title("Klasifikasi Stopkontak Listrik")
st.markdown("Prediksi status stopkontak: **Kosong**, **Terisi**, atau **Penuh**")
st.divider()


@st.cache_resource(show_spinner="Memuat model EfficientNetB0 + SVM...")
def load_effnet_svm():
    import joblib
    import tensorflow as tf
    import numpy as _np

    svm_model = joblib.load("FIX-efficientnetb0-svm_model.pkl")
    le = joblib.load("FIX-efficientnetb0-svm-label_encoder.pkl")

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
        weights="imagenet",
    )

    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=base.output,
    )

    dummy = _np.zeros((1, 224, 224, 3), dtype=_np.float32)
    feature_model.predict(dummy, verbose=0)

    return feature_model, svm_model, le


@st.cache_resource
def load_yolov5():
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    import torch

    model = torch.hub.load(
        "./yolov5",
        "custom",
        path="best.pt",
        source="local",
    )

    pathlib.PosixPath = temp

    return model


def predict_effnet_svm(pil_img):
    import tensorflow as tf

    feature_model, svm_model, le = load_effnet_svm()

    img = pil_img.convert("RGB").resize((224, 224))

    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)

    feat = feature_model.predict(arr, verbose=0)

    pred = svm_model.predict(feat)[0]
    proba = svm_model.predict_proba(feat)[0]

    label = str(pred).lower()
    conf = float(np.max(proba))

    all_conf = {
        str(cls).lower(): float(p)
        for cls, p in zip(svm_model.classes_, proba)
    }

    return label, conf, all_conf


def predict_yolov5(pil_img):
    model = load_yolov5()

    results = model(pil_img)
    df = results.pandas().xyxy[0]

    if df.empty:
        return "tidak terdeteksi", 0.0, {}

    label_map = {
        "kosong": "kosong",
        "terisi": "penuh",
        "penuh": "terisi",
    }

    top = df.loc[df["confidence"].idxmax()]

    raw_label = str(top["name"]).lower()
    label = label_map.get(raw_label, raw_label)

    conf = float(top["confidence"])

    grouped = (
        df.groupby("name")["confidence"]
        .max()
        .to_dict()
    )

    all_conf = {}

    for cls, score in grouped.items():
        raw_cls = str(cls).lower()
        fixed_cls = label_map.get(raw_cls, raw_cls)
        all_conf[fixed_cls] = float(score)

    return label, conf, all_conf


MODEL_OPTIONS = {
    "EfficientNetB0 + SVM": "effnet_svm",
    "YOLOv5s": "yolov5",
}

selected_label = st.selectbox(
    "Pilih Model",
    list(MODEL_OPTIONS.keys())
)

model_key = MODEL_OPTIONS[selected_label]

if model_key == "effnet_svm":
    load_effnet_svm()
elif model_key == "yolov5":
    load_yolov5()

st.markdown("---")

input_method = st.radio(
    "Sumber Gambar",
    ["Upload File"],
    horizontal=True,
)

pil_img = None

if input_method == "Upload File":
    uploaded = st.file_uploader(
        "Upload gambar stopkontak",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        pil_img = Image.open(uploaded)

if pil_img is not None:
    st.image(
        pil_img,
        caption="Gambar Input",
        use_container_width=True,
    )

    with st.spinner(f"Memprediksi dengan {selected_label} ..."):
        t0 = time.time()

        try:
            if model_key == "effnet_svm":
                label, conf, all_conf = predict_effnet_svm(pil_img)
            else:
                label, conf, all_conf = predict_yolov5(pil_img)

            elapsed = time.time() - t0
            success = True

        except FileNotFoundError as e:
            st.error(f"File model tidak ditemukan: {e}")
            success = False

        except Exception as e:
            st.error(f"Error: {e}")
            success = False

    if success:
        st.markdown("### Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Status Stopkontak",
                label.upper(),
            )

        with col2:
            st.metric(
                "Confidence",
                f"{conf * 100:.2f}%"
            )

        if all_conf:
            st.markdown("#### Confidence per Kelas")

            for cls, score in sorted(
                all_conf.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                st.progress(
                    min(score, 1.0),
                    text=f"{cls.upper()} — {score * 100:.2f}%"
                )

        st.caption(
            f"Waktu inferensi: {elapsed:.3f} detik | Model: {selected_label}"
        )

with st.sidebar:
    st.header("Informasi")

    st.markdown(
        """
**Kelas yang diprediksi**

- **Kosong** — Stopkontak tidak digunakan
- **Terisi** — Stopkontak terisi sebagian
- **Penuh** — Stopkontak penuh terpakai
"""
    )