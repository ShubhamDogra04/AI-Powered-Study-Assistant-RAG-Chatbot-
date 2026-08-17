import streamlit as st
import tempfile
import os
import sys
import types
import asyncio
import json
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama


# ============================================================
# RAGAS SETUP (dummy-module workaround, confirmed working
# with ragas==0.3.9)
# ============================================================

_dummy_module = types.ModuleType("vertexai")


class ChatVertexAI:
    pass


_dummy_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _dummy_module

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOllama(model="llama3.1:8b", temperature=0)


@st.cache_resource(show_spinner=False)
def load_ragas_evaluator(_llm):
    ragas_llm = LangchainLLMWrapper(_llm)
    faithfulness_metric = Faithfulness(llm=ragas_llm)
    return faithfulness_metric


FAISS_INDEX_DIR = "faiss_indexes"


def _file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()[:16]


@st.cache_resource(show_spinner="Processing document...")
def build_retrievers(file_bytes, file_extension, _embeddings):
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    index_path = os.path.join(FAISS_INDEX_DIR, _file_hash(file_bytes))

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_path)
    elif file_extension == ".txt":
        loader = TextLoader(temp_path, encoding="utf-8")
    else:
        os.remove(temp_path)
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(
            index_path, _embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(chunks, _embeddings)
        vector_store.save_local(index_path)

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    os.remove(temp_path)

    return {
        "documents": documents,
        "chunks": chunks,
        "vector_retriever": vector_retriever,
        "hybrid_retriever": hybrid_retriever,
    }


# ============================================================
# RAG HELPERS
# ============================================================

def get_context(question, retriever):
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


def generate_answer_from_context(llm, question, context, history_text=""):
    prompt = f"""You are answering questions using only the context below.

If the context genuinely does not contain the answer, say:
"I don't know based on the provided document."

Previous conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


MAX_HISTORY_TURNS = 3


async def smart_answer(question, llm, vector_retriever, hybrid_retriever, faithfulness_metric, chat_history):
    """Compares vector-only vs hybrid retrieval using RAGAS faithfulness,
    and returns whichever generated answer scored higher. Includes the
    last few turns of conversation so follow-up questions work."""

    recent_history = chat_history[-MAX_HISTORY_TURNS:]
    history_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in recent_history])

    vector_contexts = get_context(question, vector_retriever)
    vector_context = "\n\n".join(vector_contexts)

    hybrid_contexts = get_context(question, hybrid_retriever)
    hybrid_context = "\n\n".join(hybrid_contexts)

    vector_answer = generate_answer_from_context(llm, question, vector_context, history_text)
    hybrid_answer = generate_answer_from_context(llm, question, hybrid_context, history_text)

    vector_sample = SingleTurnSample(
        user_input=question,
        response=vector_answer,
        retrieved_contexts=vector_contexts
    )
    hybrid_sample = SingleTurnSample(
        user_input=question,
        response=hybrid_answer,
        retrieved_contexts=hybrid_contexts
    )

    vector_result = await faithfulness_metric.single_turn_ascore(vector_sample)
    hybrid_result = await faithfulness_metric.single_turn_ascore(hybrid_sample)

    if hybrid_result > vector_result:
        best_strategy = "Hybrid Search (BM25 + FAISS)"
        best_score = hybrid_result
        winning_answer = hybrid_answer
    else:
        best_strategy = "Vector Search (FAISS)"
        best_score = vector_result
        winning_answer = vector_answer

    return {
        "answer": winning_answer,
        "best_strategy": best_strategy,
        "best_score": best_score,
        "vector_score": vector_result,
        "hybrid_score": hybrid_result,
    }


# ============================================================
# PERSISTENT STORAGE (chat history survives app restarts)
# ============================================================

HISTORY_FILE = "chat_history.json"


def load_history_from_disk():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [tuple(entry) for entry in data]
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history_to_disk(chat_history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except OSError as e:
        st.warning(f"Could not save conversation history to disk: {e}")


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("RAG Chatbot (RAGAS-Optimized Retrieval)")
st.caption(
    "For every question, both vector-only and hybrid retrieval are tried. "
    "RAGAS scores each answer's faithfulness, and the better-grounded one is shown."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history_from_disk()

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"])

if st.session_state.chat_history:
    st.divider()
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
    st.divider()

question = st.text_input("Ask your question")

col1, col2 = st.columns([1, 1])
ask_clicked = col1.button("Ask")
clear_clicked = col2.button("Clear conversation")

if clear_clicked:
    st.session_state.chat_history = []
    save_history_to_disk(st.session_state.chat_history)
    st.rerun()

if ask_clicked:

    if uploaded_file is None:
        st.warning("Please upload a document first.")
        st.stop()

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    embeddings = load_embeddings()
    llm = load_llm()
    faithfulness_metric = load_ragas_evaluator(llm)

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    file_bytes = uploaded_file.getvalue()

    try:
        pipeline = build_retrievers(file_bytes, file_extension, embeddings)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Comparing vector vs. hybrid retrieval with RAGAS..."):
        result = asyncio.run(
            smart_answer(
                question,
                llm,
                pipeline["vector_retriever"],
                pipeline["hybrid_retriever"],
                faithfulness_metric,
                st.session_state.chat_history,
            )
        )

    st.session_state.chat_history.append((question, result["answer"]))
    save_history_to_disk(st.session_state.chat_history)

    st.success(
        f"Document processed successfully! "
        f"{len(pipeline['documents'])} pages → {len(pipeline['chunks'])} chunks"
    )

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander("Retrieval strategy details (RAGAS faithfulness scores)"):
        st.write(f"**Chosen strategy:** {result['best_strategy']}")
        st.write(f"**Vector-only faithfulness:** {result['vector_score']:.3f}")
        st.write(f"**Hybrid faithfulness:** {result['hybrid_score']:.3f}")

    st.rerun()