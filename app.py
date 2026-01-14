import streamlit as st
import cv2
from fer import FER
import numpy as np
from PIL import Image

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Emotion Detection App", layout="centered")

# ----------------- Session State -----------------
if "start" not in st.session_state:
    st.session_state.start = False

# ----------------- Welcome Page -----------------
if not st.session_state.start:
    st.title("🎭 Welcome to the Emotion Detection App!")
    st.write("""
    This app allows you to:
    - Detect emotions from uploaded images
    - Detect emotions using your webcam
    """)
    if st.button("Go to Emotion Detection"):
        st.session_state.start = True

# ----------------- Main App -----------------
if st.session_state.start:
    st.title("🎭 Real-Time Emotion Detection App")

    # ----------------- Sidebar Options -----------------
    option = st.sidebar.selectbox(
        "Select Input Mode",
        ("Upload Image", "Use Webcam")
    )

    # Initialize FER detector
    detector = FER(mtcnn=True)

    # ----------------- Function to detect emotion -----------------
    def detect_emotion(image):
        result = detector.detect_emotions(image)
        all_faces = []

        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)

            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Write dominant emotion on image
            cv2.putText(image, dominant_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            all_faces.append({
                "dominant_emotion": dominant_emotion,
                "emotions": emotions
            })

        return image, all_faces

    # ----------------- Upload Image Mode -----------------
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            annotated_image, faces = detect_emotion(image_np)
            st.image(annotated_image, channels="RGB", caption="Emotion Detection")

            # Display emotions with probabilities
            if faces:
                st.write("Detected Emotions:")
                for i, face in enumerate(faces):
                    st.write(f"**Face {i+1} (Dominant: {face['dominant_emotion']})**")
                    for emotion, score in face["emotions"].items():
                        st.write(f"- {emotion}: {score*100:.2f}%")

    # ----------------- Webcam Mode -----------------
    elif option == "Use Webcam":
        st.info("Capture your image using the webcam below")
        img_file_buffer = st.camera_input("Capture your image")

        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            image_np = np.array(image.convert('RGB'))
            annotated_image, faces = detect_emotion(image_np)
            st.image(annotated_image, channels="RGB", caption="Emotion Detection")

            # Display emotions with probabilities
            if faces:
                st.write("Detected Emotions:")
                for i, face in enumerate(faces):
                    st.write(f"**Face {i+1} (Dominant: {face['dominant_emotion']})**")
                    for emotion, score in face["emotions"].items():
                        st.write(f"- {emotion}: {score*100:.2f}%")
