import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

DB_FAISS_PATH = "vector_store/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
You are a healthcare AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, respond with:

"I could not find this information in the provided documents."

Do not guess.
Do not provide medical diagnosis.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def load_qa_chain():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        ),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": set_custom_prompt()
        }
    )

    return qa_chain