import streamlit as st
from PyPDF2 import PdfReader
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_llm_pipeline():
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    hf_pipeline = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=450,
        return_full_text=False,
        device=device,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return ChatHuggingFace(llm=llm)
    
@st.cache_resource(show_spinner=False)
def build_vector_store(chunks, _embeddings):
    return FAISS.from_texts(chunks, _embeddings)

@st.cache_resource(show_spinner=False)
def load_reranker():
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoderReranker(model=cross_encoder, top_n=3)

# PAGE CONFIG
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

has_messages = len(st.session_state.messages) > 0

if not has_messages:
    st.markdown(
        "<h1 style='text-align: center; margin-top: 15vh;'>🤖 RAG Chatbot</h1>",
        unsafe_allow_html=True
    )
else:
    st.title("🤖 RAG Chatbot")

# SIDEBAR
with st.sidebar:
    st.header("Upload PDF")
    file = st.file_uploader(
        "Upload your PDF",
        type="pdf"
    )

    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    with st.expander("⚙️ Settings & About"):
        st.markdown(
            "**RAG Chatbot** — ask questions about an uploaded PDF using "
            "hybrid search (BM25 + FAISS), cross-encoder reranking, and "
            "a local Qwen2.5 model."
        )

        if st.button("🔄 Clear resource cache"):
            st.cache_resource.clear()
            st.rerun()

# MAIN
if file is None:
    st.info("Upload a PDF in the sidebar to get started.")

if file is not None:

    # DISPLAY CHAT HISTORY
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(f"📄 View sources ({len(message['sources'])})"):
                    for i, source in enumerate(message["sources"], 1):
                        with st.container(border=True):
                            label = f"Source {i}" + (" · most relevant" if i == 1 else "")
                            st.caption(label)
                            sentences = re.split(r'(?<=[.!?])\s+', source.strip())
                            formatted = "\n\n".join(s.strip() for s in sentences if s.strip())
                            st.markdown(formatted)

    with st.spinner("📄 Processing PDF..."):
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
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'(\w) -(\w)', r'\1-\2', text)

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
        embeddings = load_embeddings()

        # VECTOR STORE
        vector_store = build_vector_store(chunks, embeddings)

    # SIDEBAR FILE INFO
    with st.sidebar:
        st.success(f"Uploaded: **{file.name}**\n\n{len(pdf_reader.pages)} pages · {len(chunks)} chunks")

    # USER QUESTION
    user_question = st.chat_input(
        "Ask your question..."
    )

    if user_question:

        # USER MESSAGE
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("Thinking..."):
            # LLM
            llm = load_llm_pipeline()

            # RETRIEVER (hybrid: BM25 keyword search + FAISS semantic search)
            bm25_retriever = BM25Retriever.from_texts(chunks)
            bm25_retriever.k = 10

            faiss_retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 10,
                    "fetch_k": 30,
                    "lambda_mult": 0.7
                }
            )

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.4, 0.6]
            )

            # RERANKER (cross-encoder narrows hybrid candidates down to the best 3)
            reranker = load_reranker()
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=hybrid_retriever
            )

            # HISTORY-AWARE RETRIEVER (rewrites follow-up questions using chat history)
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given the chat history and the latest user question, "
                           "rewrite it as a standalone question that can be understood "
                           "without the chat history. Do not answer the question, just "
                           "reformulate it if needed, otherwise return it unchanged."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            # PROMPT
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant answering questions about an "
                           "uploaded PDF. Answer only using the provided context. Give "
                           "point-wise answers using plain sentences or bullet points, "
                           "not code-comment style (no lines starting with #) unless the "
                           "answer itself is a code snippet from the context. "
                           "You must reuse the exact table names, column names, and "
                           "values from the context. Do not invent new table names, "
                           "column names, or example data, even if they seem plausible. "
                           "If the context contains an example, quote or closely paraphrase "
                           "it using its exact identifiers. Stop once you have covered what "
                           "the context provides. Do not continue with unrelated content "
                           "from other topics. "
                           "If the answer is not in the context, say \"Kindly give the "
                           "feedback\" instead of guessing."),
                MessagesPlaceholder("chat_history"),
                ("human", "Context:\n{context}\n\nQuestion:\n{input}")
            ])

            # QA CHAIN
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

            # Convert prior session_state messages into LangChain message objects
            # (exclude the current question, which was just appended above)
            chat_history_messages = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history_messages.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history_messages.append(AIMessage(content=msg["content"]))

            #  RESPONSE
            result = chain.invoke({
                "input": user_question,
                "chat_history": chat_history_messages
            })
            response = result["answer"]
            sources = [doc.page_content for doc in result["context"]]

        # ASSISTANT MESSAGE
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
        with st.chat_message("assistant"):

            st.write(response)

            with st.expander(f"📄 View sources ({len(sources)})"):
                for i, source in enumerate(sources, 1):
                    with st.container(border=True):
                        st.caption(f"Source {i}")
                        st.markdown(source.strip().replace("\n", " "))

            st.divider()

            st.write("### Was this helpful?")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Yes", key=f"yes_{len(st.session_state.messages)}"):
                    st.toast("Thanks for the feedback!")

            with col2:
                if st.button("👎 No", key=f"no_{len(st.session_state.messages)}"):
                    st.toast("We will try to help you better.")
                    