import os
from os import path
from time import sleep

from glob import glob  
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

def create_chain(api_key):
    langchain_documents = []
    docs = find_ext(".", "pdf")
    waiting_time = 30
    max_doc_arr_size = 500
    for i, textbook in enumerate(docs):
        loader = PDFMinerLoader(textbook)
        print("Loading document {}...".format(i + 1))
        pages = loader.load()
        langchain_documents += pages

    vector_store = create_vector_store(langchain_documents[0])

    for i, doc in enumerate(langchain_documents[1:]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
        document_boi = Document(
            page_content = doc.page_content,
            metatdata={
                "source": "local"
            }
        )
        splitDocs = splitter.split_documents([document_boi])
        num_docs = len(splitDocs)
        print("Adding chunk {} to vector store...".format(i + 1))
        if num_docs <= max_doc_arr_size:
            vector_store.add_documents(splitDocs)
            wait(waiting_time)
        else:
            arr_index = 0
            subchunk_number = 1
            while arr_index < num_docs:
                print("Adding subchunk {} to vector store...".format(subchunk_number))
                if arr_index + max_doc_arr_size > num_docs:
                    vector_store.add_documents(splitDocs[arr_index:num_docs])
                else:
                    vector_store.add_documents(splitDocs[arr_index:arr_index + max_doc_arr_size])
                arr_index += max_doc_arr_size
                subchunk_number += 1
                wait(waiting_time)
                

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5,
        verbose=True,
        max_tokens=4096,
        api_key=api_key
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a cognitive behavioral therapist. More context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever = vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )
    retrieval_chain = create_retrieval_chain(history_aware_retriever, chain)
    return retrieval_chain

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents([docs], embedding=embedding)
    return vector_store

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    try:
        return response["answer"]
    except Exception as e:
        print("Unable to process chat: {}".format(e))
        return None

def find_ext(dr, ext):
    return glob(path.join(dr,"*.{}".format(ext)))

def wait(waiting_time):
    print("Sleeping for {} seconds due to OpenAI API limits...".format(waiting_time))
    sleep(waiting_time)

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('api_key')
    
    chain = create_chain(api_key)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant: ", response)
