import streamlit as st
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient
import time

st.set_page_config(page_title="Talk to Your Syllabus AI", page_icon="📘")
st.title("Talk to Your Syllabus AI")

# Primary + fallback models (free tier friendly on Hugging Face)
primary = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=st.secrets["HF_TOKEN"]
)

backup = InferenceClient(
    model="google/flan-t5-large",
    token=st.secrets["HF_TOKEN"]
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

def generate_answer(prompt):
    # Try primary model
    try:
        return primary.text_generation(prompt, max_new_tokens=200, temperature=0.3)
    except Exception:
        # Small wait then retry primary once
        time.sleep(5)
        try:
            return primary.text_generation(prompt, max_new_tokens=200, temperature=0.3)
        except Exception:
            # Fallback to backup model
            return backup.text_generation(prompt, max_new_tokens=200, temperature=0.3)

if question and text:
    prompt = f"""
You are a helpful teaching assistant. Answer the question based ONLY on the syllabus text below.

Syllabus:
{text[:1500]}

Question:
{question}

Answer clearly and concisely:
"""

    with st.spinner("Thinking..."):
        try:
            response = generate_answer(prompt)
            st.write(response)
        except Exception:
            st.error("AI is under heavy load right now. Please try again in a moment.")
