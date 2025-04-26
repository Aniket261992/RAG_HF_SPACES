import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from cassandra.io.asyncioreactor import AsyncioConnection
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

import os

import os

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
ASTRA_DB_TOKEN = os.getenv('ASTRA_DB_TOKEN')

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)

if "astra_vector_index" not in st.session_state:
    loader = PyPDFLoader("/home/user/app/budget_speech.pdf",mode="single")
    docs = loader.load()
    raw_text = [doc.page_content for doc in docs]
    full_text = "/n/n".join(raw_text)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-3-large")

    cassio.init(token=ASTRA_DB_TOKEN,database_id=ASTRA_DB_ID)

    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="qa_budget_pdf",
        session=None,
        keyspace=None
    )

    spliter = RecursiveCharacterTextSplitter(separators="\n",chunk_size = 600, chunk_overlap = 60)
    split_text = spliter.split_text(full_text)

    astra_vector_store.add_texts(split_text)

    st.session_state.astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

st.title("This is a Streamlit RAG QA Chatbot using Cassandra")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"I am a QnA chatbot ready to answer your questions about budget 2025"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

user_input = st.text_input("Ask a question")

if user_input:
    with st.spinner("Searching the budget pdf"):
        st.session_state.messages.append({"role":"user","content":user_input})  
        response = st.session_state.astra_vector_index.query(user_input,llm=st.session_state.llm).strip()
        st.session_state.messages.append({"role":"assistant","content":response})
        st.success(response)  



