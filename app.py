import streamlit as st
import os

from langchain_core.messages import HumanMessage, AIMessage

from src.utils import (
    initialize_session_state,
    setup_directories,
    clear_data,
    UPLOAD_DIR
)

from src.document_loader import load_documents
from src.chunking import get_text_chunks
from src.vector_store import create_and_save_vector_store
from src.llm import test_groq_connection
from src.rag_pipeline import get_rag_chain
from audio_recorder_streamlit import audio_recorder
from src.voice import transcribe_audio, generate_audio_response


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Ventryx AI - Company Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# INITIALIZE APP
# --------------------------------------------------

initialize_session_state()
setup_directories()


# --------------------------------------------------
# ENSURE SESSION STATE EXISTS
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("Ventryx AI - Company Knowledge Assistant")

st.caption(
    "Ask questions about your company's documents using AI-powered RAG."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("Ventryx AI")

    st.caption(
        "Your Company's Knowledge, One Conversation Away."
    )

    st.divider()


    # ==================================================
    # KNOWLEDGE BASE
    # ==================================================

    st.header("📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Company Documents",
        type="pdf",
        accept_multiple_files=True
    )


    if st.button(
        "🚀 Process Documents",
        type="primary"
    ):

        if not uploaded_files:

            st.warning(
                "⚠️ Please upload at least one PDF."
            )

        else:

            with st.spinner("Processing documents..."):
                try:
                    clear_data()
                    st.session_state.chat_history = []
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                    st.info("📖 Reading documents...")
                    documents = load_documents(UPLOAD_DIR)
                    
                    if not documents:
                        st.error("⚠️ No readable text was found in the documents.")
                        st.session_state.documents_processed = False
                    else:
                        st.info("✂️ Creating text chunks...")
                        chunks = get_text_chunks(documents)
                        
                        st.info("🧠 Creating embeddings...")
                        create_and_save_vector_store(chunks)
                        
                        st.session_state.documents_processed = True
                        st.success("✅ Knowledge Base Ready")
                        st.info("💬 You can now start chatting.")
                        
                except Exception as e:
                    st.error(f"❌ Document processing failed: {e}")
                    st.session_state.documents_processed = False
                    
    st.divider()


    # ==================================================
    # AI CONFIGURATION
    # ==================================================

    st.header("🤖 AI Configuration")

    st.caption(
        "Groq LLM is configured securely through "
        "environment variables."
    )


    if st.button(
        "🔌 Test AI Connection"
    ):

        with st.spinner(
            "Testing Groq connection..."
        ):

            success, message = test_groq_connection()

            if success:

                st.success(
                    message
                )

            else:

                st.error(
                    message
                )


    st.divider()


    # ==================================================
    # CHAT CONTROLS
    # ==================================================

    st.header("💬 Chat")


    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.chat_history = []
        if "last_audio" in st.session_state:
            del st.session_state["last_audio"]

        st.rerun()

    st.header("🎤 Voice Input")
    audio_bytes = audio_recorder(
        text="Click to speak",
        recording_color="#e83e8c",
        neutral_color="#6c757d",
        icon_name="microphone",
        icon_size="2x",
        key="voice_input"
    )


# --------------------------------------------------
# DOCUMENT STATUS
# --------------------------------------------------

if not st.session_state.documents_processed:

    st.info(
        "📚 Upload your company documents from the sidebar "
        "and click 'Process Documents' to get started."
    )


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.chat_history:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )


    with st.chat_message(role):

        st.markdown(
            content
        )
        
        audio = message.get("audio")
        if audio:
            st.audio(audio, format="audio/mp3")


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

typed_question = st.chat_input(
    "Ask anything about your company..."
)

user_question = typed_question

if audio_bytes and not typed_question:
    if "last_audio" not in st.session_state or st.session_state.last_audio != audio_bytes:
        with st.spinner("Transcribing audio..."):
            transcribed_text = transcribe_audio(audio_bytes)
            if transcribed_text:
                user_question = transcribed_text
                st.session_state.last_audio = audio_bytes
                st.toast(f"You said: {transcribed_text}")


# --------------------------------------------------
# PROCESS USER QUESTION
# --------------------------------------------------

if user_question:

    # ==================================================
    # SAVE USER MESSAGE
    # ==================================================

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # ==================================================
    # DISPLAY USER MESSAGE
    # ==================================================

    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # ==================================================
    # CHECK DOCUMENT STATUS
    # ==================================================

    if not st.session_state.documents_processed:

        error_message = (
            "⚠️ Please process the documents "
            "before asking questions."
        )


        with st.chat_message("assistant"):

            st.error(
                error_message
            )


        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": error_message
            }
        )


    # ==================================================
    # GENERATE RESPONSE
    # ==================================================

    else:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                try:
                    rag_chain = get_rag_chain()
                    previous_messages = st.session_state.chat_history[:-1]
                    langchain_history = []
                    
                    for message in previous_messages:
                        role = message.get("role")
                        content = message.get("content", "")
                        
                        if role == "user":
                            langchain_history.append(HumanMessage(content=content))
                        elif role == "assistant":
                            langchain_history.append(AIMessage(content=content))
                            
                    response = rag_chain.invoke({
                        "input": user_question,
                        "chat_history": langchain_history
                    })
                    
                    answer = response.get("answer", "I couldn't generate an answer.")
                    st.markdown(answer)
                    
                    audio_response = generate_audio_response(answer)
                    if audio_response:
                        st.audio(audio_response, format="audio/mp3")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "audio": audio_response
                    })
                    
                except Exception as e:
                    error_message = f"❌ An error occurred during generation: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_message
                    })
