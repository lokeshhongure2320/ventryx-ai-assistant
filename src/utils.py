import os
import streamlit as st
import shutil

UPLOAD_DIR = "data/uploads"
VECTOR_STORE_DIR = "data/vector_store"

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = "llama3-8b-8192"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

def clear_data():
    """Clear uploads and vector store data."""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
    setup_directories()
