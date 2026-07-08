"""
Retrieval evaluation harness.

Compares the original baseline retriever (plain FAISS + MMR) against
the current hybrid (BM25 + FAISS) + cross-encoder reranked retriever,
using a hand-curated set of question -> expected-keyword pairs from
sample.pdf. Measures hit rate: did the expected content actually get
retrieved in the top-k chunks?
"""

import re
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# TEST SET: (question, expected substring in a correctly retrieved chunk)
TEST_CASES = [
    ("What is a data dictionary?", "mini database management system"),
    ("Explain foreign key with example", "referential integrity constrain"),
    ("What is specialization in database design?", "top-down process"),
    ("What is generalization?", "bottom-up process"),
    ("Difference between primary key and unique key", "allows null value"),
    ("Explain types of joins", "Equi join"),
    ("Explain NOT NULL and CHECK constraint", "Business Constrain"),
    ("What is data independence?", "Physical data independence"),
    ("Explain three-tier architecture", "ANSI-SPARC"),
    ("Explain types of keys with example", "subset of super key"),
]


def load_and_chunk(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\n+', '\n', text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )

    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'(\w) -(\w)', r'\1-\2', text)
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 40]


def build_baseline_retriever(chunks, embeddings):
    """Original approach: plain FAISS + MMR, k=3."""
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.7}
    )


def build_hybrid_reranked_retriever(chunks, embeddings):
    """Current approach: BM25 + FAISS, then cross-encoder rerank to top 3."""
    vector_store = FAISS.from_texts(chunks, embeddings)

    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 10

    faiss_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7}
    )

    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid
    )


def evaluate(retriever, test_cases, label):
    print(f"\n=== {label} ===")
    hits = 0
    for question, expected_phrase in test_cases:
        docs = retriever.invoke(question)
        found = any(
            expected_phrase.lower() in doc.page_content.lower()
            for doc in docs
        )
        status = "HIT " if found else "MISS"
        print(f"[{status}] {question}")
        if found:
            hits += 1

    hit_rate = hits / len(test_cases) * 100
    print(f"\n{label} hit rate: {hits}/{len(test_cases)} ({hit_rate:.0f}%)")
    return hit_rate


if __name__ == "__main__":
    print("Loading and chunking PDF...")
    chunks = load_and_chunk("sample.pdf")
    print(f"Total chunks: {len(chunks)}")

    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building baseline retriever...")
    baseline = build_baseline_retriever(chunks, embeddings)

    print("Building hybrid + reranked retriever...")
    improved = build_hybrid_reranked_retriever(chunks, embeddings)

    baseline_rate = evaluate(baseline, TEST_CASES, "BASELINE (plain FAISS + MMR)")
    improved_rate = evaluate(improved, TEST_CASES, "HYBRID + RERANKED")

    print("\n" + "=" * 50)
    print(f"SUMMARY: baseline {baseline_rate:.0f}% -> hybrid+reranked {improved_rate:.0f}%")
    print("=" * 50)
