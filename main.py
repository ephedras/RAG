import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
import time

def load_model():
    # Define the LLM Model
    n_gpu_layers = -1
    n_batch = 500
    context_len = 2048
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="Model/zephyr-7b-beta.Q4_0.gguf",
        temperature=0.1,
        max_tokens=2000,
        n_gpu_layers=n_gpu_layers,
        n_ctx=context_len,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    return llm

def load_data():
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
    vector_store = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)
    return vector_store

# Configure web app
st.set_page_config(
    page_title="PDF RAG",
    page_icon="assets/logo_icon_clear.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image('assets/full_logo_clear.png', width=200)

# Initialize model and vector store if not already done
if 'llm' not in st.session_state:
    st.session_state.llm = load_model()
    st.session_state.vector_store = load_data()
    st.session_state.rag_pipe = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type='stuff',
        retriever=st.session_state.vector_store.as_retriever(k=1, fetch_k=6)
    )

tabinfo, tababout = st.sidebar.tabs(["INFO", "ABOUT"])
with tabinfo:
    st.info('''This is a part of work designed for Retrieval Augmented Generation (RAG) using open-source 
                models''')
with tababout:
    st.warning('Designed by:\n\nEphedras, Coimbatore\n\nver. 1.2\n\nDate: 27/03/2024', icon="🔥")

st.title("Question Answering System")

start = end = 0
prompt = st.chat_input("Say something")

response = ""
msgCont = st.container(height=400)

if prompt:
    with st.status("Progress...", expanded=True) as status:
        start = time.time()
        st.write(time.strftime('%H:%M:%S') + " : Processing prompt")
        msgCont.chat_message("user").write(prompt)
        st.write(time.strftime('%H:%M:%S') + " : Processing the result...")
        # Display assistant response in chat message container
        with msgCont.chat_message("assistant"):
            ref_doc = st.session_state.vector_store.similarity_search(prompt, fetch_k=6)
            output = st.markdown(st.session_state.rag_pipe(prompt)['result'])
            for doc in ref_doc:
                st.write("Reference :")
                st.info(f"Content: {doc.page_content}")
                st.warning(f"Reference: {doc.metadata}")

        end = time.time()
        st.write(time.strftime('%H:%M:%S') + " : Completed process!")
        status.update(label=f"Response time :  {(end - start):.2f}s!", state="complete", expanded=False)
