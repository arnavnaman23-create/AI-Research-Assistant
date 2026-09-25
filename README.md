# AI-Research-Assistant
An AI-powered research assistant that uses RAG, FAISS, Groq, and LangGraph to answer questions, explain concepts, and extract key findings from research papers.
# AI Research Assistant

An AI-powered research assistant that helps users explore research papers using Retrieval-Augmented Generation (RAG), FAISS, Groq, and LangGraph.

## Features

- 📄 Research paper analysis
- 💬 Question answering using RAG
- 🧠 Explain complex concepts in simple language
- 🔍 Extract key findings
- 📚 Source and page-based retrieval
- 🔀 LangGraph-based query routing
- ⚡ Groq-powered LLM responses
- 🔎 FAISS vector search

## Technologies Used

- Python
- Streamlit
- Groq
- LangGraph
- FAISS
- Sentence Transformers
- PyPDF
- NumPy
- Pandas

## Architecture

Research Paper  
↓  
PDF Processing  
↓  
Text Chunking  
↓  
Sentence Transformer Embeddings  
↓  
FAISS Vector Database  
↓  
User Query  
↓  
LangGraph Query Routing  
↓  
Relevant Context Retrieval  
↓  
Groq LLM  
↓  
Grounded Answer

## Project Structure

```text
AI-Research-Assistant/
├── app.py
├── config.py
├── rag.py
├── graph.py
├── requirements.txt
├── documents/
└── vector_store/
