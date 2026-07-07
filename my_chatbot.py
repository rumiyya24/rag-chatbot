import streamlit as st
from PyPDF2 import PdfReader
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# PAGE CONFIG
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")

# SIDEBAR
with st.sidebar:
    st.header("Upload PDF")
    file = st.file_uploader(
        "Upload your PDF",
        type="pdf"
    )

# FEEDBACK MODAL
@st.dialog("Feedback")
def feedback_dialog(message):
    st.write(message)

# MAIN
if file is not None:

    # READ PDF
    pdf_reader = PdfReader(file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # CLEAN TEXT
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'', '', text)

    # CHUNKING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_text(text)

    chunks = [
        chunk.strip()
        for chunk in chunks
        if len(chunk.strip()) > 40
    ]

    #  EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # VECTOR STORE
    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    # USER QUESTION
    user_question = st.chat_input(
        "Ask your question..."
    )

    if user_question:

        # USER MESSAGE
        with st.chat_message("user"):
            st.write(user_question)

        # QUERY ENHANCEMENT
        query = f"topic: {user_question}"

        # RETRIEVER
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )

        # LLM
        hf_pipeline = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_new_tokens=256,
            return_full_text=False
        )

        llm = HuggingFacePipeline(
            pipeline=hf_pipeline
        )

        # PROMPT
        prompt_template = """
Rules:
- Give point-wise answers
- Upload Q&A type pdf file
- If answer is not found:
  "Kindly give the feedback"
Context:
{context}
Question:
{input}
Answer:
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "input"]
        )

        # QA CHAIN
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, combine_docs_chain)

        #  RESPONSE
        result = chain.invoke({"input": user_question})
        response = result["answer"]

        # ASSISTANT MESSAGE
        with st.chat_message("assistant"):

            st.write(response)

            st.divider()

            st.write("### Was this helpful?")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Yes"):
                    feedback_dialog(
                        "Thanks for the feedback! 😊"
                    )

            with col2:
                if st.button("👎 No"):
                    feedback_dialog(
                        "We will try to help you better."
                    )