from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
load_dotenv() 
from langchain_core.documents import Document

docs = [
    Document(page_content = "Python is a high-level, interpreted programming language known for its simple syntax and readability. It is widely used in web development, data science, artificial intelligence, automation, and software development. Python supports object-oriented, functional, and procedural programming styles. Popular frameworks and libraries include Django, Flask, NumPy, and TensorFlow.",
    metaData={"python"}),
        Document(page_content = "pandas is a Python library used for data manipulation and analysis through structures like DataFrames and Series, making it useful for handling large datasets efficiently.",
    metaData={"python"}),
        Document(page_content = "Artificial Neural Network is a machine learning model inspired by the human brain that learns patterns from data using interconnected neurons and is widely used in deep learning, image recognition, and natural language processing.",
    metaData={"python"})
]

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma-db"
)

result = vectorstore.similarity_search("what is used for data analysis?",k=2)

for r in result:
    print(r.page_content)
    print("  ----  ")
    print(r.metadata)

retriver = vectorstore.as_retriever() 

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)