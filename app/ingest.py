from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.logger import logger

DATA_PATH = "data/"
DB_FAISS_PATH = "vector_store/db_faiss"

logger.info("Loading PDFs")
logger.info("Creating embeddings")
logger.info("Saving FAISS database")


def ingest_documents():

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    text_chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        text_chunks,
        embedding_model
    )

    db.save_local(DB_FAISS_PATH)

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(text_chunks),
        "status": "success"
    }
    