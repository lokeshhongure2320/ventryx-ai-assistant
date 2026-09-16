from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from src.llm import get_llm
from src.vector_store import load_vector_store


# --------------------------------------------------
# 1. Prompt for understanding follow-up questions
# --------------------------------------------------

CONTEXTUALIZE_PROMPT = """
You are a question reformulation assistant.

Given the conversation history and the user's latest question,
rewrite the latest question into a standalone question.

The standalone question must be understandable without the
conversation history.

Do NOT answer the question.

Only return the rewritten question.

Conversation History:
{chat_history}

Latest User Question:
{input}
"""


# --------------------------------------------------
# 2. Prompt for answering using company documents
# --------------------------------------------------

QA_PROMPT = """
You are Ventryx AI, the company knowledge assistant
for Ventryx Technology.

Answer the user's question using ONLY the information contained
in the retrieved company documents.

You can use the conversation history to understand references such as:

- it
- they
- this
- that
- the company
- previous year
- above
- mentioned earlier

However, the actual factual answer MUST come only from the
retrieved company documents.

If the answer cannot be found in the provided context, say:

"I couldn't find this information in the uploaded company documents."

Never invent company policies, employee information, procedures,
salaries, rules, financial information, or other company facts.

Keep the answer clear and concise.

Conversation History:
{chat_history}

Retrieved Context:
{context}

User Question:
{input}
"""


# --------------------------------------------------
# 3. Create RAG Chain
# --------------------------------------------------

def get_rag_chain():
    """
    Create conversational RAG pipeline.
    """

    # --------------------------------------------------
    # Load LLM
    # --------------------------------------------------

    llm = get_llm()

    # --------------------------------------------------
    # Load Vector Store
    # --------------------------------------------------

    vector_store = load_vector_store()

    if vector_store is None:
        raise ValueError(
            "Vector store not found. Please process documents first."
        )

    # --------------------------------------------------
    # Create Retriever
    # --------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    # --------------------------------------------------
    # Prompt for Follow-up Questions
    # --------------------------------------------------

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                CONTEXTUALIZE_PROMPT,
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )

    # --------------------------------------------------
    # History-Aware Retriever
    # --------------------------------------------------

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt,
    )

    # --------------------------------------------------
    # Prompt for Final Answer
    # --------------------------------------------------

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                QA_PROMPT,
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )

    # --------------------------------------------------
    # Document Answering Chain
    # --------------------------------------------------

    document_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )

    # --------------------------------------------------
    # Final Conversational RAG Chain
    # --------------------------------------------------

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    return retrieval_chain