import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.title {
    font-size: 36px;
    font-weight: bold;
    color: #1f2937;
}
.subtitle {
    font-size: 18px;
    color: #6b7280;
}
.box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.stButton>button {
    width: 100%;
    height: 45px;
    font-size: 16px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="mansoorh/news-classifier"
    )

classifier = load_model()

labels = ["🌍 World", "⚽ Sports", "💼 Business", "💻 Sci/Tech"]

# Header
st.markdown('<div class="title">📰 News Topic Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Write a news headline from any category below:</div>',
    unsafe_allow_html=True
)

# Category info
st.info("Categories: World 🌍 | Sports ⚽ | Business 💼 | Sci/Tech 💻")

st.markdown("---")

# Input section
st.markdown('<div class="box">', unsafe_allow_html=True)

text = st.text_area(
    "Enter your headline:",
    placeholder="e.g., Government announces new economic policy..."
)

predict_btn = st.button("Predict")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if predict_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter a headline.")
    else:
        result = classifier(text)[0]
        label_id = int(result["label"].split("_")[-1])
        confidence = result["score"]

        st.markdown("### ✅ Prediction Result")
        st.success(f"{labels[label_id]}")

        # Confidence display
        st.markdown("### 📊 Confidence")
        st.progress(float(confidence))
        st.write(f"{round(confidence * 100, 2)}% confidence")

# Footer
st.markdown("---")
st.markdown(
    "<center>AI News Classifier</center>",
    unsafe_allow_html=True
)