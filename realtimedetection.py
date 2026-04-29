import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2
import random
import time

st.set_page_config(page_title="TF Hub Object Detection", layout="wide")
st.title("🔥 TensorFlow Hub Object Detection App")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return hub.load(
        "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    ).signatures["default"]

model = load_model()

colorcodes = {}

# ---------------- DRAW BOX ----------------
def drawbox(image, ymin, xmin, ymax, xmax, name, color):
    h, w, _ = image.shape

    left, right = int(xmin * w), int(xmax * w)
    top, bottom = int(ymin * h), int(ymax * h)

    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    cv2.putText(image, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ---------------- DETECTION ----------------
def run_detection(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = tf.image.convert_image_dtype(img_rgb, tf.float32)[tf.newaxis, ...]

    result = model(img_tensor)
    result = {k: v.numpy() for k, v in result.items()}

    boxes = result["detection_boxes"]
    classes = result["detection_class_entities"]
    scores = result["detection_scores"]

    for i in range(min(6, len(scores))):
        if scores[i] < 0.1:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        label = classes[i].decode("utf-8")
        score = int(scores[i] * 100)

        if label not in colorcodes:
            colorcodes[label] = (
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255)
            )

        drawbox(frame, ymin, xmin, ymax, xmax,
                f"{label}:{score}%", colorcodes[label])

    return frame

# ---------------- SIDEBAR ----------------
option = st.sidebar.radio(
    "Choose Input Type",
    ["📷 Webcam", "🖼️ Image Upload", "🎥 Video Upload"]
)

# =========================================================
# 📷 WEBCAM
# =========================================================
if option == "📷 Webcam":
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not found")
                break

            frame = cv2.resize(frame, (640, 480))
            frame = run_detection(frame)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()

# =========================================================
# 🖼️ IMAGE UPLOAD
# =========================================================
elif option == "🖼️ Image Upload":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        result_img = run_detection(image)

        st.image(result_img, channels="BGR", caption="Detected Image")

# =========================================================
# 🎥 VIDEO UPLOAD
# =========================================================
elif option == "🎥 Video Upload":
    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if file:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        cap = cv2.VideoCapture(temp_path)

        frame_placeholder = st.empty()

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % 2 == 0:
                frame = cv2.resize(frame, (640, 480))
                frame = run_detection(frame)

                frame_placeholder.image(frame, channels="BGR")

            time.sleep(0.01)

        cap.release()