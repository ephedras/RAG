# import
from langchain_community.document_loaders import TextLoader,PyPDFDirectoryLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

def datacreator_chroma(persist_directory = 'db'):

    # load the document and split it into chunks
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load()
    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    db = Chroma.from_documents(texts, embedding_function, persist_directory=persist_directory)
    

    return db

def datacreator_faiss(persist_directory = 'db'):

    # load the document and split it into chunks
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load()
    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into faiss
    db = FAISS.from_documents(texts, embedding_function)
    db.save_local("faiss_index")

    return db

def dataloader_faiss(faiss_db = 'faiss_index'):
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(faiss_db, embedding_function)

# # print results
# print(docs[0].page_content)
    
if __name__ == "__main__":
    datacreator_faiss()