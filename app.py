import os, pickle
import cv2, numpy as np, streamlit as st
import mediapipe as mp
import streamlit.components.v1 as components

st.set_page_config(page_title="SignTalk - Sign Language Translator", page_icon="🤟")
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils

# thumb, index, middle, ring, pinky  (1 = extended)
GESTURES = {
    (1, 1, 1, 1, 1): "HELLO",
    (0, 0, 0, 0, 0): "YES",
    (1, 0, 0, 0, 0): "GOOD",
    (0, 1, 1, 0, 0): "PEACE",
    (1, 1, 0, 0, 1): "I LOVE YOU",
    (1, 0, 0, 0, 1): "CALL ME",
    (0, 1, 0, 0, 0): "WAIT",
    (1, 1, 0, 0, 0): "YOU",
    (0, 1, 1, 1, 0): "WATER",
    (0, 1, 1, 1, 1): "THANK YOU",
}


def dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)


def finger_states(lm):
    thumb = int(dist(lm[4], lm[17]) > dist(lm[3], lm[17]))  # works for either hand
    others = [int(lm[t].y < lm[p].y) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
    return (thumb, *others)


def rule_based(lm):
    s = finger_states(lm)
    if dist(lm[4], lm[8]) < 0.05 and s[2:] == (1, 1, 1):
        return "OK"
    return GESTURES.get(s, "?")


def features(lm):
    pts = np.array([[p.x, p.y, p.z] for p in lm])
    pts -= pts[0]
    return (pts / (np.abs(pts).max() + 1e-6)).flatten()


@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            return pickle.load(f)


model = load_model()
st.title("🤟 SignTalk: Sign Language Translator")
st.caption("AI Innovation Challenge | Hand-landmark detection (MediaPipe) + gesture classification")

modes = ["Gesture words (rule-based)"] + (["Alphabet (trained ML model)"] if model else [])
mode = st.radio("Recognition mode", modes, horizontal=True)

if "sentence" not in st.session_state:
    st.session_state.sentence = []

src = st.radio("Input", ["Camera", "Upload image"], horizontal=True)
img_file = st.camera_input("Show a sign") if src == "Camera" else st.file_uploader("Image", type=["jpg", "jpeg", "png"])

if img_file:
    img = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        res = hands.process(rgb)
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(rgb, lm, mp_hands.HAND_CONNECTIONS)
        pred = rule_based(lm.landmark) if mode.startswith("Gesture") else model.predict([features(lm.landmark)])[0]
        st.image(rgb, caption="Detected hand landmarks", use_container_width=True)
        st.success(f"Translation: **{pred}**")
        if pred != "?" and st.button("➕ Add to sentence"):
            st.session_state.sentence.append(pred)
    else:
        st.image(rgb, use_container_width=True)
        st.warning("No hand detected. Use good light and keep your palm in frame.")

text = " ".join(st.session_state.sentence)
st.subheader("Sentence")
st.write(text or "_(empty)_")
c1, c2 = st.columns(2)
if c1.button("🔊 Speak"):
    components.html(
        f"<script>speechSynthesis.speak(new SpeechSynthesisUtterance({text!r}));</script>", height=0
    )
if c2.button("🗑️ Clear"):
    st.session_state.sentence = []
    st.rerun()

with st.expander("Supported gestures"):
    st.write(", ".join(sorted(set(GESTURES.values()) | {"OK"})))
