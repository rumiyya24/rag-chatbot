# RAG Chatbot using LangChain + Streamlit

A Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, LangChain, FAISS, HuggingFace, and a local Qwen2.5 model. Upload a PDF and ask questions about its content.

This fork extends the original project with hybrid search, reranking, conversational memory, source citation, and a measured retrieval evaluation harness. See [What's New](#whats-new-in-this-fork) below.

## Features

- Upload PDF documents
- Hybrid search: BM25 keyword matching + FAISS semantic search
- Cross-encoder reranking for improved retrieval precision
- Conversational memory (follow-up questions resolve using chat history)
- Source citation -- view the exact retrieved chunks behind each answer
- Retrieval-Augmented Generation with a local, free Qwen2.5 model
- Cached models/vector store for fast repeat queries
- Feedback buttons
- Retrieval evaluation harness with measured hit-rate metrics

## Tech Stack

- Python, Streamlit
- LangChain 1.x (`langchain-classic`, `langchain-huggingface`, `langchain-community`)
- FAISS (semantic search) + BM25 (`rank_bm25`, keyword search)
- Cross-encoder reranking (`sentence-transformers`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Qwen2.5-1.5B-Instruct (local, via HuggingFace Transformers)
- PyPDF2

## Project Structure

    project/
    │
    ├── my_chatbot.py          # Main Streamlit app
    ├── eval_retrieval.py       # Retrieval evaluation harness (dev tool)
    ├── requirements.txt
    ├── README.md
    ├── .streamlit/
    │   └── config.toml         # Disables dev-mode hotkey conflicts
    ├── .gitignore
    └── sample.pdf

## Installation

### 1. Clone repository
```bash
git clone https://github.com/rumiyya24/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

Note: if you also have conda installed, run `conda deactivate` first -- mixing conda's base environment with a venv can cause the wrong package versions to be picked up.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

This installs pinned versions of the LangChain family of packages. LangChain went through a major restructuring; `RetrievalQA` and related chains moved into a separate `langchain-classic` package. This project uses the current API.

## Run the application
```bash
streamlit run my_chatbot.py
```

Upload `sample.pdf` (or your own PDF) in the sidebar, then ask a question in the chat box.

## How It Works

1. **Upload PDF** -- user uploads a PDF document
2. **Text Extraction** -- PyPDF2 extracts text from the PDF
3. **Text Cleaning** -- normalizes whitespace and rejoins words split by PDF line-wrapping (e.g. "top -down" becomes "top-down")
4. **Chunking** -- LangChain splits text into overlapping chunks
5. **Embedding + Indexing** -- chunks are embedded (`all-MiniLM-L6-v2`) and indexed in FAISS
6. **Hybrid Retrieval** -- a query retrieves candidates via both BM25 (keyword) and FAISS (semantic) search, merged via Reciprocal Rank Fusion
7. **Reranking** -- a cross-encoder rescoring pass narrows the candidates down to the most relevant chunks
8. **History-Aware Retrieval** -- follow-up questions are rewritten into standalone questions using chat history before retrieval
9. **Generation** -- Qwen2.5-1.5B-Instruct generates an answer, constrained to only use retrieved content
10. **Source Display** -- the actual chunks used are shown to the user for verification

## Measured Retrieval Improvement

Using `eval_retrieval.py`, a 10-question hand-curated test set against `sample.pdf`:

| Retriever | Hit Rate |
|---|---|
| Baseline (plain FAISS + MMR, original approach) | 60% |
| Hybrid search (BM25 + FAISS) + cross-encoder reranking | 70% |

Run it yourself:
```bash
python eval_retrieval.py
```

This is a retrieval-only metric (does the correct chunk get retrieved), not an LLM-judged faithfulness score -- see Known Limitations below for why.

## What's New in This Fork

Compared to the original project, this fork adds:
- Migration to LangChain 1.x (the original code targeted an older, now-incompatible API)
- Fixed a model/task mismatch that caused garbled/echoed answers
- Hybrid search (BM25 + FAISS) instead of semantic-only retrieval
- Cross-encoder reranking
- Conversational memory for follow-up questions
- A PDF text-cleaning fix that substantially improved retrieval accuracy on its own
- A retrieval evaluation harness with measured before/after numbers
- Source citation UI
- Caching (previously the model and vector store were rebuilt on every single question)
- General UI polish (loading states, empty state, lighter feedback confirmation)

## Known Limitations

- **Small-model faithfulness**: even after prompt tuning and greedy decoding, the local 1.5B model occasionally generates plausible-sounding details (e.g. mislabeling a key type) not present in the source document. This is a known limitation of small local models rather than a retrieval bug -- a larger model would likely reduce this further.
- **No LLM-judged evaluation**: `eval_retrieval.py` measures retrieval hit rate only, not answer faithfulness/relevancy (RAGAS-style metrics), since those require an LLM-as-judge -- by default a paid OpenAI model, or a local model too small to be a reliable judge.
- Best results are achieved with clean, well-structured PDFs; scanned or heavily formatted PDFs may still retrieve poorly.

## Future Improvements

- LLM-judged faithfulness evaluation (with a stronger judge model)
- Multi-PDF support
- Streaming responses
- Production deployment

## Authors

- Original project: Shivani Chauhan
- Fork improvements: Rumiyya ([@rumiyya24](https://github.com/rumiyya24))

Contributions are welcome. Feel free to open a Pull Request.
