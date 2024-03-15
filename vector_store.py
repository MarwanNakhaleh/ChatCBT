import os
import sys

from dotenv import load_dotenv

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

def create_vector_store(collection, index, docs):
    embedding = OpenAIEmbeddings()
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=docs, 
        embedding=embedding,
        collection=collection,
        index_name=index
    )
    return vector_store

def get_db(connection_str, db_name):
    client = MongoClient(connection_str)
    db = client[db_name]
    return db

if __name__ == "__main__":
    load_dotenv()
    client = MongoClient(os.getenv('mongodb_conn_str'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
        sys.exit(0)

        
