import time 
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import  UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,CacheBackedEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


#.streamlit 폴더 생성 >> secrets.toml >> OPEN_API_KEY 지정 하기 
#gitingore 에 해당폴더 추가 

st.set_page_config(
    page_title="Document GPT",
    page_icon="🤖",
)

#streamlit 이 사용하기 위한 캐쉬기능을 위해 embeddings , files 폴더 .cache 폴더 하위에 생성 
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    st.write(file_content,file_path)
    with open(file_path,"wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}") 
    splitter = CharacterTextSplitter.from_tiktoken_encoder(  
        separator="\n\n",
        chunk_size = 600,
        chunk_overlap = 50,

    )


    loader = UnstructuredFileLoader("./files/document.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder,cache_dir,
    )

    vectorstores = FAISS.from_documents(docs,cached_embeddings) 
  
    retriever = vectorstores.as_retriever()
    return retriever


st.title("Document GPT")

st.markdown("""
Welcome, 
             
Use this chatbot to ask a question to an AI for your local file .             

""")

file = st.file_uploader("Upload file txt or pdf excel",type=["txt","pdf","xlsx"])

if file:
   retriever = embed_file(file)
   s = retriever.invoke("winston")
   s