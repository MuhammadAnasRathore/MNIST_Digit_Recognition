import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢", layout="centered")

# ------------------ FRONT-END VISUAL UPGRADE (NO LOGIC CHANGES) ------------------
st.markdown("""
<style>
/* ---- Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;500;700&display=swap');

:root{
  --bg0:#05060a;
  --bg1:#070a14;
  --neonC:#41f1ff;
  --neonB:#5b7cff;
  --neonP:#b04bff;
  --glass: rgba(12, 16, 30, 0.55);
  --glass2: rgba(12, 16, 30, 0.35);
  --stroke: rgba(120, 200, 255, 0.25);
  --stroke2: rgba(176, 75, 255, 0.22);
  --text: rgba(235, 245, 255, 0.92);
  --muted: rgba(235, 245, 255, 0.70);
  --shadow: 0 22px 70px rgba(0,0,0,0.55);
}

/* ---- App background: deep space gradient + glow ---- */
.stApp{
  background:
    radial-gradient(1200px 700px at 70% 10%, rgba(176,75,255,0.25), transparent 60%),
    radial-gradient(900px 500px at 20% 30%, rgba(65,241,255,0.18), transparent 55%),
    radial-gradient(700px 450px at 60% 70%, rgba(91,124,255,0.15), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
  overflow-x: hidden;
}

/* ---- Subtle animated particles ---- */
.stApp::before{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events:none;
  background-image:
    radial-gradient(circle at 10% 20%, rgba(255,255,255,0.09) 0 1px, transparent 2px),
    radial-gradient(circle at 80% 30%, rgba(255,255,255,0.08) 0 1px, transparent 2px),
    radial-gradient(circle at 40% 70%, rgba(255,255,255,0.06) 0 1px, transparent 2px),
    radial-gradient(circle at 70% 80%, rgba(255,255,255,0.07) 0 1px, transparent 2px);
  background-size: 260px 260px, 320px 320px, 380px 380px, 420px 420px;
  animation: dustFloat 12s linear infinite;
  opacity: 0.55;
  mix-blend-mode: screen;
}
@keyframes dustFloat{
  from{ transform: translate3d(0,0,0); }
  to{ transform: translate3d(-60px, 40px, 0); }
}

/* ---- Holographic grid floor ---- */
.stApp::after{
  content:"";
  position: fixed;
  left:0; right:0; bottom:-10vh;
  height: 52vh;
  pointer-events:none;
  background:
    linear-gradient(to top, rgba(65,241,255,0.14), transparent 65%),
    repeating-linear-gradient(90deg, rgba(65,241,255,0.14) 0 1px, transparent 1px 45px),
    repeating-linear-gradient(0deg, rgba(176,75,255,0.10) 0 1px, transparent 1px 45px);
  transform: perspective(900px) rotateX(72deg);
  transform-origin: bottom center;
  filter: blur(0.2px);
  opacity: 0.55;
}

/* ---- 3D floating digits in background (CSS only) ---- */
.mnist-bg-digits{
  position: fixed;
  inset: 0;
  pointer-events:none;
  z-index: 0;
  opacity: 0.38;
  filter: drop-shadow(0 0 18px rgba(65,241,255,0.25));
}
.mnist-bg-digits span{
  position:absolute;
  font-family: Orbitron, sans-serif;
  font-weight: 800;
  color: rgba(170, 210, 255, 0.12);
  text-shadow:
    0 0 18px rgba(65,241,255,0.18),
    0 0 40px rgba(176,75,255,0.12);
  transform-style: preserve-3d;
  animation: spinFloat 10s ease-in-out infinite;
}
@keyframes spinFloat{
  0%{ transform: translate3d(0,0,0) rotateY(0deg) rotateX(6deg); }
  50%{ transform: translate3d(0,-16px,0) rotateY(180deg) rotateX(-6deg); }
  100%{ transform: translate3d(0,0,0) rotateY(360deg) rotateX(6deg); }
}

/* ---- Ensure main content sits above background ---- */
.block-container{ position: relative; z-index: 2; padding-top: 2.2rem; max-width: 980px; }

/* ---- Title + subtitle ---- */
h1, h2, h3{
  font-family: Orbitron, sans-serif !important;
  letter-spacing: 0.5px;
}
h1{
  text-align:center;
  font-weight: 800 !important;
  color: rgba(240,250,255,0.95);
  text-shadow: 0 0 22px rgba(65,241,255,0.22), 0 0 42px rgba(176,75,255,0.18);
  margin-bottom: 0.25rem !important;
}
div[data-testid="stMarkdownContainer"] p{
  font-family: Rajdhani, sans-serif;
  color: var(--muted);
  font-size: 1.05rem;
  text-align: center;
}

/* -------------------- Upload: make it single-purpose -------------------- */
/* Hide drag/drop helper text + file metadata line */
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] *{
  display:none !important;
}

/* Upload container as clean panel */
div[data-testid="stFileUploader"]{
  background: linear-gradient(180deg, rgba(10,14,28,0.55), rgba(10,14,28,0.30));
  border: 1px solid rgba(65,241,255,0.20);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 0 0 1px rgba(176,75,255,0.10) inset, 0 20px 55px rgba(0,0,0,0.40);
  backdrop-filter: blur(14px);
  position: relative;
}

/* Remove dashed dropzone look and turn it into a centered "button only" surface */
div[data-testid="stFileUploader"] section{
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
}

/* Hide filename/details row after upload (keeps uploader functional, removes clutter) */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]{
  display:none !important;
}

/* ---- Upload label styling (clean) ---- */
div[data-testid="stFileUploader"] label{
  font-family: Orbitron, sans-serif !important;
  color: rgba(235,245,255,0.92) !important;
  text-align: center;
  width: 100%;
  display:block;
  margin-bottom: 10px;
}

/* -------------------- Buttons: Predict + Upload must match -------------------- */
/* Predict button already styled via .stButton>button.
   Force the file uploader internal "Browse files" button to match exactly. */
div[data-testid="stFileUploader"] button{
  width: 100% !important;
  font-family: Orbitron, sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px;
  color: rgba(5,6,10,0.95) !important;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.85rem 1rem !important;
  background: linear-gradient(90deg, rgba(65,241,255,1), rgba(176,75,255,1)) !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.08) inset, 0 14px 40px rgba(65,241,255,0.15) !important;
  position: relative !important;
  overflow: hidden !important;
  transform: translateZ(0) !important;
}

/* Pulse overlay for upload button (match Predict) */
div[data-testid="stFileUploader"] button::after{
  content:"";
  position:absolute;
  inset:-40%;
  background: radial-gradient(circle, rgba(255,255,255,0.65), transparent 45%);
  transform: translateX(-60%) rotate(12deg);
  opacity: 0.30;
  animation: pulseGlow 2.6s ease-in-out infinite;
  pointer-events:none;
}

/* Hover */
div[data-testid="stFileUploader"] button:hover{
  filter: brightness(1.08);
  box-shadow: 0 0 18px rgba(65,241,255,0.22), 0 0 48px rgba(176,75,255,0.18) !important;
}

/* Predict button (keep your existing style) */
.stButton>button{
  width: 100%;
  font-family: Orbitron, sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px;
  color: rgba(5,6,10,0.95) !important;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.85rem 1rem !important;
  background: linear-gradient(90deg, rgba(65,241,255,1), rgba(176,75,255,1)) !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.08) inset, 0 14px 40px rgba(65,241,255,0.15);
  position: relative;
  overflow: hidden;
  transform: translateZ(0);
}
.stButton>button::after{
  content:"";
  position:absolute;
  inset:-40%;
  background: radial-gradient(circle, rgba(255,255,255,0.65), transparent 45%);
  transform: translateX(-60%) rotate(12deg);
  opacity: 0.30;
  animation: pulseGlow 2.6s ease-in-out infinite;
}
@keyframes pulseGlow{
  0%,100%{ transform: translateX(-55%) rotate(12deg); opacity:0.22; }
  50%{ transform: translateX(35%) rotate(12deg); opacity:0.40; }
}
.stButton>button:hover{
  filter: brightness(1.08);
  box-shadow: 0 0 18px rgba(65,241,255,0.22), 0 0 48px rgba(176,75,255,0.18);
}

/* -------------------- Tabs hover/active fix (no white/grey flash) -------------------- */
div[data-testid="stTabs"] button{
  font-family: Orbitron, sans-serif !important;
  color: rgba(235,245,255,0.72) !important;
  background: rgba(12,16,30,0.18) !important;
  border-radius: 12px 12px 0 0 !important;
  border: 1px solid rgba(120,200,255,0.14) !important;
  transition: all 220ms ease !important;
}
div[data-testid="stTabs"] button:hover{
  color: rgba(240,250,255,0.95) !important;
  background: rgba(65,241,255,0.10) !important;
  box-shadow: 0 0 18px rgba(65,241,255,0.12), 0 0 36px rgba(176,75,255,0.10) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"]{
  color: rgba(240,250,255,0.96) !important;
  background: linear-gradient(90deg, rgba(65,241,255,0.14), rgba(176,75,255,0.12)) !important;
  border-bottom: 1px solid rgba(12,16,30,0.0) !important;
  box-shadow: 0 0 22px rgba(65,241,255,0.14), 0 0 44px rgba(176,75,255,0.12) !important;
}

/* -------------------- KPI cards for Predicted Digit + Confidence -------------------- */
.kpi-wrap{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 14px;
}
.kpi-card{
  background: linear-gradient(180deg, rgba(12, 16, 30, 0.62), rgba(12, 16, 30, 0.28));
  border: 1px solid rgba(65,241,255,0.18);
  border-radius: 18px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.35);
  backdrop-filter: blur(14px);
  padding: 16px 16px;
  text-align: center;
  position: relative;
  overflow:hidden;
}
.kpi-card::before{
  content:"";
  position:absolute;
  inset:-2px;
  border-radius: 18px;
  background: conic-gradient(from 180deg, rgba(65,241,255,0.0), rgba(65,241,255,0.22), rgba(176,75,255,0.18), rgba(91,124,255,0.18), rgba(65,241,255,0.0));
  filter: blur(12px);
  opacity: 0.55;
  z-index: 0;
  animation: neonSweep 6s linear infinite;
}
.kpi-label{
  position: relative;
  z-index: 1;
  font-family: Rajdhani, sans-serif;
  color: rgba(235,245,255,0.70);
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.kpi-value{
  position: relative;
  z-index: 1;
  font-family: Orbitron, sans-serif;
  font-weight: 800;
  font-size: 2.4rem;
  line-height: 1.1;
  color: rgba(240,250,255,0.97);
  text-shadow: 0 0 18px rgba(65,241,255,0.20), 0 0 34px rgba(176,75,255,0.14);
}

/* -------------------- Raw probability chart container polish -------------------- */
.chart-title{
  font-family: Orbitron, sans-serif;
  text-align:center;
  font-weight: 800;
  color: rgba(240,250,255,0.95);
  letter-spacing: 1px;
  margin: 18px 0 10px 0;
  text-shadow: 0 0 18px rgba(65,241,255,0.16), 0 0 36px rgba(176,75,255,0.12);
}
.vega-wrap{
  background: linear-gradient(180deg, rgba(12,16,30,0.55), rgba(12,16,30,0.22));
  border: 1px solid rgba(176,75,255,0.16);
  border-radius: 16px;
  padding: 12px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
}

/* ---- Footer ---- */
.mnist-footer{
  position: fixed;
  left: 0; right: 0; bottom: 18px;
  z-index: 3;
  text-align: center;
  font-family: Orbitron, sans-serif;
  font-weight: 700;
  letter-spacing: 2.2px;
  color: rgba(255, 232, 170, 0.92);
  text-shadow:
    0 0 10px rgba(255, 210, 90, 0.22),
    0 0 26px rgba(65,241,255,0.18),
    0 0 48px rgba(176,75,255,0.12);
  opacity: 0.95;
  pointer-events:none;
}

/* Header transparent */
header[data-testid="stHeader"]{ background: transparent; }
</style>

<!-- Background digits -->
<div class="mnist-bg-digits">
  <span style="top:10%; left:8%; font-size:92px; animation-duration:11s;">0</span>
  <span style="top:18%; left:72%; font-size:120px; animation-duration:13s;">7</span>
  <span style="top:55%; left:12%; font-size:110px; animation-duration:12s;">3</span>
  <span style="top:62%; left:78%; font-size:98px; animation-duration:10s;">9</span>
  <span style="top:32%; left:42%; font-size:150px; animation-duration:14s;">5</span>
</div>

<!-- Footer -->
<div class="mnist-footer">Made by Muhammad Anas Rathore</div>
""", unsafe_allow_html=True)
# ------------------ END FRONT-END VISUAL UPGRADE ------------------


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

# ---------------- UI ----------------
st.title(" MNIST DIGIT RECOGNITION ")
st.markdown("Upload a clear image of a digit (0–9)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Preprocess
from PIL import ImageFilter

def preprocess(image):
    img = image.convert("L")
    img = img.resize((28, 28))

    # 🔥 enhancement
    img = img.filter(ImageFilter.SHARPEN)

    arr = np.array(img).astype("float32") / 255.0

    # normalize contrast
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    return arr.reshape(1, 28, 28, 1), img



# ---------------- MAIN ----------------
if uploaded_file:

    st.success("Image uploaded successfully")

    image = Image.open(uploaded_file)

    processed, preview = preprocess(image)

    # ---------------- PREDICTION ----------------
    if st.button("Predict Digit"):

        preds = model.predict(processed)[0]
        digit = np.argmax(preds)
        confidence = preds[digit] * 100

        # Premium KPI cards (digit + confidence)
        st.markdown(f"""
        <div class="kpi-wrap">
          <div class="kpi-card">
            <div class="kpi-label">Predicted Digit</div>
            <div class="kpi-value">{digit}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Confidence</div>
            <div class="kpi-value">{confidence:.2f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Raw Probability chart (premium, dark, neon)
        st.markdown('<div class="chart-title">PROBABILITY GRAPH</div>', unsafe_allow_html=True)

        # Build a modern bar chart with Altair (replaces plain Streamlit bar_chart)
        try:
            import pandas as pd
            import altair as alt

            df = pd.DataFrame({
                "Digit": [str(i) for i in range(10)],
                "Probability": [float(p) for p in preds]
            })

            chart = alt.Chart(df).mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6
            ).encode(
                x=alt.X("Digit:N", axis=alt.Axis(labelColor="#cfe9ff", title=None, labelFont="Orbitron", labelFontSize=12)),
                y=alt.Y("Probability:Q", axis=alt.Axis(labelColor="#cfe9ff", title=None, grid=True, gridColor="rgba(120,200,255,0.12)")),
                color=alt.Color(
                    "Probability:Q",
                    scale=alt.Scale(range=["#41f1ff", "#5b7cff", "#b04bff"]),
                    legend=None
                ),
                tooltip=[alt.Tooltip("Digit:N"), alt.Tooltip("Probability:Q", format=".4f")]
            ).properties(
                height=320
            ).configure_view(
                stroke=None
            ).configure(
                background="rgba(0,0,0,0)"
            ).configure_axis(
                domainColor="rgba(120,200,255,0.20)",
                tickColor="rgba(120,200,255,0.20)"
            )

            st.markdown('<div class="vega-wrap">', unsafe_allow_html=True)
            st.altair_chart(chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception:
            # fallback (keeps app working even if altair/pandas not available)
            st.bar_chart({str(i): float(p) for i, p in enumerate(preds)})

else:
    st.warning("Upload a digit image to start")