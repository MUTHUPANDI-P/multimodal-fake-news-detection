import streamlit as st
from groq import Groq
from langdetect import detect
from PIL import Image
import cv2
import numpy as np
import pytesseract
import requests
import base64
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ================== SETUP ==================
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
load_dotenv()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(
    page_title="Multimodal & Multilingual Fake News Detection",
    layout="wide"
)

# ================== GLOBAL LIGHT UI ==================
st.markdown("""
<style>
html, body, .stApp {
    background-color:#f8fafc;
    color:#0f172a;
}

.block-container {
    padding-top:2rem;
    padding-bottom:2rem;
}

.card {
    background:white;
    padding:28px;
    border-radius:16px;
    box-shadow:0 10px 25px rgba(0,0,0,0.08);
    margin:25px auto;
    max-width:1100px;
}

.fake {
    background:#fee2e2;
    color:#991b1b;
    padding:16px;
    border-radius:12px;
    font-size:22px;
    font-weight:800;
    text-align:center;
}

.real {
    background:#dcfce7;
    color:#166534;
    padding:16px;
    border-radius:12px;
    font-size:22px;
    font-weight:800;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================

def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = load_image_base64("logo.png")

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:center; gap:12px;">
    <img src="data:image/png;base64,{logo}" width="85"/>
    <h1 style="margin:0;">Multimodal & Multilingual Fake News Detection</h1>
</div>
<p style="text-align:center; opacity:0.7;">
    fake news detection using NLP, OCR, and web scraping.
</p>
""", unsafe_allow_html=True)


st.divider()

# ================== HELPER FUNCTIONS ==================
def is_valid_news(text):
    text = text.strip().lower()

    if len(text.split()) < 5:
        return False

    greetings = ["hi", "hello", "hey", "ok", "test"]
    if text in greetings:
        return False

    return True


def detect_language_safe(text):
    try:
        if len(text.split()) < 5:
            return "Not enough text"

        lang_code = detect(text)
        lang_map = {
            "en": "English",
            "ta": "Tamil",
            "hi": "Hindi",
            "te": "Telugu",
            "kn": "Kannada",
            "ml": "Malayalam"
        }
        return lang_map.get(lang_code, lang_code.upper())
    except:
        return "Unknown"


def call_groq(news_text):
    prompt = f"""
You are a STRICT fake news detection system.

VERY IMPORTANT RULES (NO EXCEPTIONS):

1. Claims that contradict basic science, biology, physics, or medicine MUST be marked FAKE.
   - Examples: viruses spreading through radio waves, magnets, towers, or apps.

2. Mobile towers, 5G, WiFi, radio waves, Bluetooth CANNOT spread viruses or cause infections → ALWAYS FAKE.

3. Medical claims MUST be supported by recognized scientific organizations.
   - If a claim lacks evidence from WHO, CDC, ICMR, or peer-reviewed journals → FAKE.

4. Herbal, home, or traditional remedies claiming to cure ALL diseases, cancer, COVID, diabetes, or HIV → FAKE.

5. Government-related claims (schemes, bans, laws, announcements):
   - If no official government source is mentioned → FAKE.

6. Sensational or fear-based language:
   - Words like “shocking”, “secret”, “doctors hide”, “media won’t tell you” → FAKE.

7. Social media forwards, WhatsApp messages, or anonymous “experts say” claims → FAKE.

8. Old or recycled news presented as new → FAKE.

9. Financial scams:
   - “Get rich quick”, “guaranteed returns”, “free money from government” → FAKE.

10. Deepfake images/videos or AI-generated content presented as real events → FAKE.

11. If the claim cannot be verified using reliable public sources → FAKE.

12. SATIRE OR PARODY NEWS:
   - Content from satire websites (e.g., The Onion) or humorous/exaggerated articles
     that are NOT intended to be factual → ALWAYS FAKE.

13. DO NOT be neutral.
   - NEVER say “needs more research” or “may be true”.
   - You MUST decide: REAL or FAKE.

OUTPUT RULES:
- Choose ONLY ONE verdict: REAL or FAKE
- Follow scientific and factual consensus
- Be concise, clear, and confident


FINAL VERDICT: REAL or FAKE
Explanation: short (2–3 lines)
Verification Tips: how users can verify

News:
\"\"\"{news_text}\"\"\""""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a professional fact-checker."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content


def extract_text_from_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    return pytesseract.image_to_string(gray, lang="eng+tam+hin+tel+kan+mal")


def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text().split())
        return text[:4000]
    except:
        return ""

# ================== INPUT SECTION ==================
st.markdown("## 📥 News Input")

input_type = st.radio(
    "Choose input type",
    ["📝 Text", "🌐 URL", "🖼️ Image"],
    horizontal=True
)

news_text = ""

if input_type == "📝 Text":
    news_text = st.text_area(
        "Paste news content",
        height=200,
        placeholder="Paste news article text here..."
    )

elif input_type == "🌐 URL":
    url = st.text_input("Enter news article URL")
    if url:
        news_text = extract_text_from_url(url)

elif input_type == "🖼️ Image":
    uploaded_image = st.file_uploader(
        "Upload news image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_image:
       image = Image.open(uploaded_image)

       col1, col2, col3 = st.columns([1, 2, 1])
       with col2:
           st.image(image, use_container_width=True, caption="Uploaded Image")

       news_text = extract_text_from_image(image)


# ================== ANALYSIS ==================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze = st.button("🔍 Analyze News", use_container_width=True)
if analyze:

    # ✅ VALIDATION CHECK
    if not is_valid_news(news_text):
        st.warning("⚠️ Please provide valid news content (not greetings or very short text).")
        st.stop()

    with st.spinner("🤖 Analyzing..."):
        try:
            lang = detect_language_safe(news_text)
            result = call_groq(news_text)

            st.markdown("## 📊 Analysis Result")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Detected Language**")
                st.info(lang)

            with col2:
                if "FAKE" in result.upper():
                    st.markdown("<div class='fake'>🚨 FAKE NEWS</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='real'>✅ REAL NEWS</div>", unsafe_allow_html=True)

            st.markdown("### 📌 Explanation")
            st.write(result)

            st.markdown("### 🔍 Verification Tips")
            st.markdown("""
- Check trusted news websites  
- Verify using official sources  
- Avoid sensational claims  
- Cross-check using Google News  
""")

        except Exception as e:
            st.error(f"❌ API Error: {e}")

