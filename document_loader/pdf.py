from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import TokenTextSplitter,RecursiveCharacterTextSplitter
data = PyPDFLoader("L6.pdf")


docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap =10
)

chunks =  splitter.split_documents(docs)
print(chunks[1].page_content)
