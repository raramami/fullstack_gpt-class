from typing import Dict, List
from uuid import UUID 
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import  UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,CacheBackedEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from pathlib import Path


#.streamlit 폴더 생성 >> secrets.toml >> OPEN_API_KEY 지정 하기 
#gitingore 에 해당폴더 추가 

st.set_page_config(
    page_title="Document GPT",
    page_icon="🤖",
)



class ChatCallBackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message,"ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token 
        self.message_box.markdown(self.message)

llm = ChatOpenAI( 
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallBackHandler()
        ],
    )

#streamlit 이 사용하기 위한 캐쉬기능을 위해 embeddings , files 폴더 .cache 폴더 하위에 생성 
@st.cache_data(show_spinner="Embedding the file ...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    #st.write(file_content,file_path)
    # with open(file_path,"wb") as f:
    #     f.write(file_content)
    Path("./.cache/files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}") 
    splitter = CharacterTextSplitter.from_tiktoken_encoder(  
        separator="\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder,cache_dir,
    )

    vectorstores = FAISS.from_documents(docs,cached_embeddings) 
  
    retriever = vectorstores.as_retriever()
    return retriever

def save_message(message,role):
    st.session_state["messages"].append({"message":message,"role":role})

def send_message(message,role,save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message,role)
       

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"],message["role"],save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    Answer the question using ONLY the following context . If you don't know the answer 
    Just say you don't know. Don't make anything up . 
    Context: {context}
    """),
    ("human","{question}")
]
)

st.title("Document GPT")

st.markdown("""
Welcome, 
             
Use this chatbot to ask a question to an AI for your local file .    
Upload your file in the sidebar          

""")

with st.sidebar:
    file = st.file_uploader("Upload file txt or pdf excel",type=["txt","pdf","xlsx"])

if file:
   retriever = embed_file(file)
  
   send_message("I'm ready ! Ask Away ","ai",save=False)
   paint_history()
   message = st.chat_input("Ask anything about your file ...")

   if message:
       send_message(message,"human")
       chain = (
           {
           "context": retriever | RunnableLambda(format_docs) ,
           "question": RunnablePassthrough()
       } | prompt | llm 
       )

       with st.chat_message("ai"):
            response = chain.invoke(message)
        
       #st.write(chain)
else:
    st.session_state["messages"] = []
