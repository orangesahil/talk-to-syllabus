import streamlit as st
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Talk to Your Syllabus AI", page_icon="📘")
st.title("Talk to Your Syllabus AI")

# Use a free, stable model on Hugging Face
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
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
            response = client.text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.3,
            )
            st.write(response)
        except Exception:
            st.error("AI service is busy right now. Please try again in 20–30 seconds.")
