from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """Return the HuggingFace embeddings model."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
