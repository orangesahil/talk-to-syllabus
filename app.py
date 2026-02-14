import streamlit as st
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient
import time

st.set_page_config(page_title="Talk to Your Syllabus AI", page_icon="📘")
st.title("Talk to Your Syllabus AI")

HF_TOKEN = st.secrets["HF_TOKEN"]

# Fallback models (tries next if one is busy)
MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

def get_client(model):
    return InferenceClient(model=model, token=HF_TOKEN)

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
    last_error = None

    for model in MODELS:
        try:
            client = get_client(model)
            return client.text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.3,
            )
        except Exception as e:
            last_error = e
            time.sleep(1)

    raise last_error

if question and text:
    prompt = f"""
You are a helpful teaching assistant. Answer the question based ONLY on the syllabus text below.

Syllabus:
{text[:1800]}

Question:
{question}

Answer clearly and concisely:
"""

    with st.spinner("Thinking..."):
        try:
            response = ask_ai(prompt)
            st.write(response)
        except Exception:
            st.error("🚨 AI servers are busy right now. Please try again in 30–60 seconds.")

