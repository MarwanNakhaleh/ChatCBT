import sys
import os

from glob import glob  
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.llms import GPT4All

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

from vector_store import VectorStore

def create_chain(vs, model_path):
    langchain_documents = []
    split_texts = []
    callbacks = [StreamingStdOutCallbackHandler()]

    if vs.collection_is_empty():
        docs = find_ext(".", "pdf")
        for i, textbook in enumerate(docs):
            loader = PDFMinerLoader(textbook)
            print("Loading document {}...".format(i + 1))
            pages = loader.load()
            langchain_documents += pages

        for doc in langchain_documents:
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
            split_texts += splitDocs

        vector_store = vs.create_vector_store("default", split_texts)
    else:
        print("data is already in collection, searching now...")
        vector_store = vs.create_vector_search()

    llm = GPT4All(
        model=model_path,
        backend="gptj",
        callbacks=callbacks,
        verbose=True,
    )
    template = """Context: {context}
    
    Question: {input}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)
    
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

def process_chat(chain, question, context, chat_history, vector_store):
    print("processing chat...")
    # docs = vector_store.similarity_search(question)
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
        "context": context
    })
    try:
        return response["answer"]
    except Exception as e:
        print("Unable to process chat: {}".format(e))
        return None

def find_ext(dr, ext):
    return glob(os.path.join(dr,"*.{}".format(ext)))


if __name__ == "__main__":
    context = input("What is the role of the GPT?: ")
    if 'exit' in context.lower():
        sys.exit(0)
    load_dotenv()
    database_name = "db"
    collection_name = "default"
    vs = VectorStore(os.getenv("mongodb_conn_str"), database_name, collection_name)
    
    chain = create_chain(vs, ("./openassistant-llama2-13b-orca-8k-3319.Q4_K_S.gguf"))

    chat_history = []

    while True:
        user_input = input("What is your question?: ")
        if 'exit' in user_input.lower():
            break
        response = process_chat(chain, user_input, context, chat_history, vs.vector_store)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
