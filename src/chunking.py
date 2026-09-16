import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_chunks(documents, chunk_size=1000, chunk_overlap=150):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add unique chunk_id to each chunk
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        
    return chunks
