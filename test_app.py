import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("RAG Chatbot")


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"]
)


# ============================================================
# QUESTION
# ============================================================

question = st.text_input(
    "Ask your question"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("Ask"):

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if uploaded_file is None:
        st.warning("Please upload a document first.")
        st.stop()

    # --------------------------------------------------------
    # Check question
    # --------------------------------------------------------

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()


    # ========================================================
    # 1. SAVE UPLOADED FILE TEMPORARILY
    # ========================================================

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        temp_path = temp_file.name


    # ========================================================
    # 2. LOAD DOCUMENT
    # ========================================================

    if file_extension == ".pdf":

        loader = PyPDFLoader(temp_path)

    elif file_extension == ".txt":

        loader = TextLoader(
            temp_path,
            encoding="utf-8"
        )

    else:

        st.error("Unsupported file type.")
        os.remove(temp_path)
        st.stop()


    documents = loader.load()


    # ========================================================
    # 3. SPLIT DOCUMENT INTO CHUNKS
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(
        documents
    )


    # ========================================================
    # 4. CREATE EMBEDDINGS
    # ========================================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # ========================================================
    # 5. CREATE FAISS VECTOR STORE
    # ========================================================

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )


    # ========================================================
    # 6. VECTOR RETRIEVER
    # ========================================================

    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )


    # ========================================================
    # 7. BM25 RETRIEVER
    # ========================================================

    bm25_retriever = BM25Retriever.from_documents(
        chunks
    )

    bm25_retriever.k = 3


    # ========================================================
    # 8. HYBRID RETRIEVER
    # ========================================================

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever
        ],
        weights=[
            0.4,
            0.6
        ]
    )


    # ========================================================
    # 9. RETRIEVE RELEVANT DOCUMENTS
    # ========================================================

    retrieved_docs = hybrid_retriever.invoke(
        question
    )


    # ========================================================
    # 10. CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    # ========================================================
    # 11. LOAD LLAMA 3.1
    # ========================================================

    llm = OllamaLLM(
         model="llama3.1:8b"
    )


    # ========================================================
    # 12. CREATE PROMPT
    # ========================================================

    prompt = f"""
You are a helpful RAG chatbot.

Answer the question using ONLY the context provided below.

If the answer is not present in the context,
say:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""


    # ========================================================
    # 13. GENERATE ANSWER
    # ========================================================

    with st.spinner("Generating answer..."):

        response = llm.invoke(
            prompt
        )


    # ========================================================
    # 14. DISPLAY RESULTS
    # ========================================================

    st.success(
        f"Document processed successfully! "
        f"{len(documents)} pages → {len(chunks)} chunks"
    )

    st.subheader("Answer")

    st.write(response)


    # ========================================================
    # 15. CLEANUP
    # ========================================================

    os.remove(temp_path)