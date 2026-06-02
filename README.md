# Healthcare AI Assistant

# Overview

Healthcare AI Assistant is a Retrieval-Augmented Generation (RAG) based system that answers healthcare-related queries using information extracted from medical documents.
The system combines semantic search, vector databases, and Large Language Models (LLMs) to provide context-aware and grounded responses while minimizing hallucinations.


# Features

* Retrieval-Augmented Generation (RAG)
* PDF document ingestion
* Semantic search using FAISS
* FastAPI REST APIs
* Agent-based query routing
* Source citation support
* Dockerized deployment
* Swagger API documentation


# System Architecture

Healthcare Documents (PDF)

↓

Document Loader

↓

Text Chunking

↓

Embedding Generation (MiniLM)

↓

FAISS Vector Store

↓

Retriever

↓

Groq LLM

↓

Response Generation

↓

Answer + Sources


# Project Structure

healthcare-ai-assistant/

app/

* main.py
* rag.py
* agent.py
* ingest.py
* logger.py

data/

vector_store/

scripts/

tests/

Dockerfile

docker-compose.yml

requirements.txt

README.md


# APIs

# GET /health

Returns application health status.

# POST /ask

Accepts a healthcare-related question and returns an answer along with source references.

Example:

{
"question": "What is cancer?"
}

# POST /ingest

Loads documents, generates embeddings, and updates the FAISS vector store.

# Agent Workflow

The application uses a lightweight agentic workflow:

* Appointment-related queries are routed to scheduling tools.
* Knowledge-based queries are routed through the RAG pipeline.

Example:

Question:
"Can I book a cardiology appointment?"

↓

Appointment Tool

Question:
"What is cancer?"

↓

RAG Pipeline


# Technologies Used

* Python
* FastAPI
* LangChain
* FAISS
* Hugging Face Embeddings
* Groq
* Docker
* Streamlit (Prototype UI)

# Running Locally

Install dependencies:

pip install -r requirements.txt

Configure environment variables:

GROQ_API_KEY=your_api_key

Run:

python -m uvicorn app.main:app --reload

Swagger:

http://localhost:8000/docs

# Docker Deployment

Build:

docker compose build

Run:

docker compose up

Swagger:

http://localhost:8000/docs


# Future Improvements

* Multi-document ingestion
* Persistent database storage
* Authentication and authorization
* Appointment booking integration
* Medical knowledge graph integration
* Monitoring and analytics

# Architecture Diagram

┌─────────────────────┐
│ Medical PDF Files   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Document Chunking   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ MiniLM Embeddings   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FAISS Vector Store  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Retriever           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Groq LLM            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Answer + Sources    │
└─────────────────────┘

# Author

Omkar Ballal

B.Tech Computer Science and Design

Dr. D Y Patil School of Science and Technology
