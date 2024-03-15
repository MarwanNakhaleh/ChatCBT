import os
import sys

from dotenv import load_dotenv

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

class VectorStore:
    connection_string = ""
    mongo_client = None
    db = None
    collection = None
    db_name = ""
    collection_name = ""

    def __init__(self, connection_string, db_name, collection_name):
        self.connection_string = connection_string
        self.mongo_client = MongoClient(connection_string)
        self.db_name = db_name
        self.db = self.mongo_client[db_name]
        self.collection_name = collection_name
        self.collection = self.db[collection_name]

    def create_vector_store(self, index, docs):
        embedding = OpenAIEmbeddings()
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=docs, 
            embedding=embedding,
            collection=self.collection,
            index_name=index
        )
        return vector_store

    def create_vector_search(self):
        vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            os.getenv('mongodb_conn_str'),
            f"{self.db_name}.{self.collection_name}",
            OpenAIEmbeddings(),
            index_name="default"
        )
        return vector_search
    
    def collection_is_empty(self):
        return self.collection.count() == 0

if __name__ == "__main__":
    load_dotenv()
    vs = VectorStore(os.getenv('mongodb_conn_str'), "db", "default")
    try:
        vs.mongo_client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
        sys.exit(0)

        
