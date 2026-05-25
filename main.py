import os
import re
import unicodedata
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# =========================
# LOAD ENV
# =========================

load_dotenv()

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions from it.")

# =========================
# EMBEDDING MODEL
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LLM
# =========================

llm = ChatMistralAI(
    model="mistral-small-2506"
)

# =========================
# PROMPT TEMPLATE
# =========================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a strict PDF question-answering assistant.

        Answer ONLY from the provided context.

        Rules:
        - Do NOT use outside knowledge.
        - Do NOT guess.
        - Do NOT make up answers.
        - If the answer is not clearly present in the context,
          reply ONLY with:

          "I don't know the answer."

        Give concise answers based only on the context.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])

# PDF UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    # Create folder if not exists
    os.makedirs("uploaded_docs", exist_ok=True)

    pdf_path = os.path.join(
        "uploaded_docs",
        uploaded_file.name
    )

    # Save uploaded file
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # =========================
    # LOAD PDF
    # =========================

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # =========================
    # TEXT SPLITTING
    # =========================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    texts = []

    for chunk in chunks:

        text = chunk.page_content

        if text and isinstance(text, str):

            text = unicodedata.normalize("NFKD", text)

            text = text.encode(
                "utf-8",
                errors="ignore"
            ).decode("utf-8")

            text = re.sub(
                r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
                "",
                text
            )

            text = re.sub(r"\s+", " ", text).strip()

            if text:
                texts.append(text)

    # =========================
    # VECTOR STORE
    # =========================

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        persist_directory="chroma-db"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    st.success("Vector database created!")

    # =========================
    # CHAT INPUT
    # =========================

    query = st.text_input("Ask a question")

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)