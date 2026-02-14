import streamlit as st
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient

st.title("Talk to Your Syllabus AI")

import streamlit as st
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=st.secrets["HF_TOKEN"]
)


uploaded_file = st.file_uploader("Upload your syllabus PDF", type="pdf")

text = ""

if uploaded_file:
    reader = PdfReader(uploaded_file)
    
    for page in reader.pages:
        text += page.extract_text()

    st.success("PDF uploaded successfully")

question = st.text_input("Ask question from syllabus")

if question and text:

    prompt = f"""
    Answer based on syllabus:

    {text[:2000]}

    Question:
    {question}
    """

    response = client.text_generation(
        prompt,
        max_new_tokens=200,
    )

    st.write(response)

