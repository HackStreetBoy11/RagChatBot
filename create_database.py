import re
import unicodedata
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("document_loader/deepLearning.pdf")
docs = loader.load()

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
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            texts.append(text)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("Vector DB created successfully!")