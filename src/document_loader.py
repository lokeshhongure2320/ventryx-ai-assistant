import os
from langchain_community.document_loaders import PyMuPDFLoader

def load_documents(directory_path: str):
    """Load all PDF documents from the specified directory."""
    documents = []
    if not os.path.exists(directory_path):
        return documents

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            loader = PyMuPDFLoader(file_path)
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata["document_name"] = filename
                # PyMuPDFLoader already adds 'page' metadata (0-indexed usually). 
                # Let's ensure it's a 1-indexed page_number for presentation.
                page_num = doc.metadata.get("page", 0)
                if isinstance(page_num, int):
                    doc.metadata["page_number"] = page_num + 1
                else:
                    doc.metadata["page_number"] = page_num
            
            documents.extend(loaded_docs)
            
    return documents
