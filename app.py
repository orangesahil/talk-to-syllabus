import streamlit as st
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient
import time

st.set_page_config(page_title="Talk to Your Syllabus AI", page_icon="📘")
st.title("Talk to Your Syllabus AI")

# Load HuggingFace token safely
HF_TOKEN = st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN is missing. Please add it in Streamlit Secrets.")
    st.stop()

# Use a lighter, faster model (more reliable on free tier)
client = InferenceClient(
    model="google/flan-t5-base",
    token=HF_TOKEN,
    timeout=60
)

uploaded_file = st.file_uploader("Upload your syllabus PDF", type="pdf")

text = ""

if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    st.success("PDF uploaded successfully")

question = st.text_input("Ask question from syllabus")

def ask_ai(prompt):
    for attempt in range(3):
        try:
            return client.text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.2,
            )
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)

if question and text:
    prompt = f"""
Answer strictly from the syllabus content below.

Syllabus:
{text[:1500]}

Question:
{question}

Give a short, clear answer:
"""

    with st.spinner("Thinking..."):
        try:
            response = ask_ai(prompt)
            st.success("Answer:")
            st.write(response)
        except Exception:
            st.error("AI servers are overloaded. Try again in 30–60 seconds.")
