from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents([docs], embedding=embedding)
    return vector_store