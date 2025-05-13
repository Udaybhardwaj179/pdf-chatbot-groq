import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set Streamlit UI
st.set_page_config(page_title="Chat with PDF (Groq + LangChain)")
st.title("📄 Chat with PDF using Groq API")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Load and split PDF content into chunks
@st.cache_data
def load_and_split_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

# Store chunks in vector store (FAISS)
@st.cache_resource
def store_chunks_in_vector_store(_chunks):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(_chunks, embeddings)

# Process after file is uploaded
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and vectorize
    chunks = load_and_split_pdf("temp.pdf")
    vector_store = store_chunks_in_vector_store(chunks)

    # Setup Groq-compatible model
    llm = ChatOpenAI(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.3
    )

    # QA chain setup
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

    # User input
    user_query = st.text_input("Ask something from the PDF:")

    if user_query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(user_query)
            st.success(result)
