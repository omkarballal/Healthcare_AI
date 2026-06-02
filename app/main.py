from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import ingest_documents
from app.agent import route_question
from app.logger import logger


from app.rag import load_qa_chain

app = FastAPI()

qa_chain = load_qa_chain()

class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Healthcare AI Assistant"
    }
@app.post("/ask")
def ask(request: QuestionRequest):
    
    logger.info("Question received")
    logger.info(request.question)

    route = route_question(request.question)

    # Tool Routing
    if route["type"] == "tool":
        return {
            "question": request.question,
            "answer": route["response"]
        }

    # RAG Pipeline
    response = qa_chain.invoke(
        {"query": request.question}
    )

    sources = list(
       set(
           doc.metadata.get("source")
           for doc in response["source_documents"]
        )
    )
    
    return {
        "question": request.question,
        "answer": response["result"],
        "sources": sources
    }
    
@app.post("/ingest")
def ingest():

    result = ingest_documents()

    return result