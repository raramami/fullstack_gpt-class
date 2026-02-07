# (KR)
# 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
# 파일 업로드 및 채팅 기록을 구현합니다.
# 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
#https://fullstackgpt-class-ezouhwmuhmdihbbbfv67cx.streamlit.app/


import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import  UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from pathlib import Path


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



#streamlit 이 사용하기 위한 캐쉬기능을 위해 embeddings , files 폴더 .cache 폴더 하위에 생성 
@st.cache_data(show_spinner="Embedding the file ...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    Path("./.cache/files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}") 
    splitter = CharacterTextSplitter.from_tiktoken_encoder(  
        separator="\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )


    loader = UnstructuredFileLoader(f"{file_path}")
    docs = loader.load_and_split(text_splitter=splitter)
    embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
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


def main():
    if not openai_api_key:
        return
    
    llm = ChatOpenAI( 
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallBackHandler()
            ],
        )

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
        return
        
with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key")
    file = st.file_uploader("Upload file txt or pdf excel",type=["txt","pdf","xlsx"])

try:
    main()

except Exception as e:
    st.error("Check your OpenAI API Key or File")
    st.write(e)