import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq


## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

# Configure page
st.set_page_config(
    page_title="Healthcare AI - Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Source documents styling */
    .source-docs {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        border-left: 4px solid #000000;
        font-size: 0.9rem;
        color: #000000;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .info-title {
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    /* Input area styling */
    .input-container {
        margin-top: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #764ba2;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


DB_FAISS_PATH="vector_store/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🏥 About Healthcare AI")
        st.info(
            "Healthcare AI is an advanced AI-powered healthcare assistant that helps answer medical questions "
            "based on trusted medical knowledge. Always consult with a healthcare professional for medical advice."
        )
        
        st.markdown("### ⚙️ Settings")
        st.markdown("**Model**: Llama 3.3 70B (Groq)")
        st.markdown("**Embedding Model**: MiniLM-L6-v2")
        st.markdown("**Temperature**: 0.0 (Precise)")
        
        st.markdown("### 📊 Session Info")
        if 'messages' in st.session_state:
            st.metric("Messages", len(st.session_state.messages))
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🏥 Healthcare AI</h1>
        <p class="header-subtitle">Advanced Medical Knowledge Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    st.markdown("### 💬 Conversation")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"**👤 You**: {message['content']}", unsafe_allow_html=True)
            else:
                # Split result and source docs
                content = message['content']
                if 'Source Docs:' in content:
                    result, sources = content.split('Source Docs:', 1)
                    st.markdown(f"**🤖 Assistant**: {result.strip()}", unsafe_allow_html=True)
                    st.markdown(f"<div class='source-docs'><b>📎 Source Documents:</b>{sources}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**🤖 Assistant**: {content}", unsafe_allow_html=True)

    # Input section
    st.markdown("---")
    st.markdown("### ❓ Ask a Question")
    prompt = st.chat_input("Ask me anything about healthcare...", max_chars=500)

    if prompt:
        # Display user message
        st.markdown(f"**👤 You**: {prompt}", unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        # Progress indicator
        with st.spinner("🤔 Thinking..."):
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("❌ Failed to load the vector store. Please check the database.")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.0,
                        groq_api_key=os.environ["GROQ_API_KEY"],
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})

                result = response["result"]
                source_documents = response["source_documents"]
                
                # Format source documents nicely
                sources_text = ""
                for i, doc in enumerate(source_documents, 1):
                    sources_text += f"\n{i}. {doc.page_content[:200]}..."
                
                result_to_show = result + "\n**Source Docs:**" + sources_text
                
                # Display assistant response
                st.markdown(f"**🤖 Assistant**: {result}", unsafe_allow_html=True)
                st.markdown(f"<div class='source-docs'><b>📎 Sources ({len(source_documents)}):</b>{sources_text}</div>", unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result + "\nSource Docs:" + sources_text
                })
                
                st.success("✅ Response generated successfully!")
                
            except KeyError:
                st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("💡 Tip: Make sure the vector store is properly initialized.")


if __name__ == "__main__":
    main()