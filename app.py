import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Image CAPTCHA Solver", page_icon="🔐", layout="centered")

st.title("🔐 Image CAPTCHA Solver")
st.write("Upload a CAPTCHA image and get the predicted text.")

# =========================
# SETTINGS
# =========================
MODEL_PATH = "image_captcha_model.h5"
IMG_WIDTH = 220
IMG_HEIGHT = 70
MAX_LABEL_LENGTH = 6   # change if your labels are not 6 chars

# =========================
# CHARACTER VOCAB
# =========================
# apne training dataset ke hisab se same characters rakho
characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, mask_token=None)
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)

# =========================
# CUSTOM CTC LAYER
# =========================
class CTCLayer(tf.keras.layers.Layer):
    def call(self, y_true, y_pred):
        return y_pred

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"CTCLayer": CTCLayer}
    )
    return model

model = load_model()

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("L")
    img = np.array(image)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.medianBlur(img, 3)

    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img, image

# =========================
# DECODE PREDICTION
# =========================
def decode_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :MAX_LABEL_LENGTH]

    texts = []
    for seq in decoded:
        seq = tf.boolean_mask(seq, seq != -1)
        seq = tf.cast(seq, tf.int64)
        text = tf.strings.reduce_join(num_to_char(seq)).numpy().decode("utf-8")
        texts.append(text)
    return texts

# =========================
# UI
# =========================
uploaded_file = st.file_uploader("Upload CAPTCHA image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    processed_img, original_img = preprocess_image(uploaded_file)

    st.image(original_img, caption="Uploaded CAPTCHA", use_container_width=True)

    if st.button("Predict CAPTCHA"):
        pred = model.predict(processed_img)
        pred_text = decode_predictions(pred)[0]
        st.success(f"Predicted CAPTCHA: {pred_text}")
