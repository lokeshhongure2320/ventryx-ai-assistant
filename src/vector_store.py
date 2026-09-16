import os
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embeddings
from src.utils import VECTOR_STORE_DIR

def create_and_save_vector_store(chunks):
    """Create a FAISS vector store from document chunks and save it to disk."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)
    return vector_store

def load_vector_store():
    """Load the FAISS vector store from disk."""
    if not os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
        return None
    
    embeddings = get_embeddings()
    # allow_dangerous_deserialization is needed since we trust the local file we just created
    vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vector_store
