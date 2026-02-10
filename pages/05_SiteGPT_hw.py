# (KR)
# Cloudflare 공식문서(https://developers.cloudflare.com)를 위한 SiteGPT 버전을 만드세요.
# 챗봇은 아래 프로덕트의 문서에 대한 질문에 답변할 수 있어야 합니다:
# AI Gateway
# Cloudflare Vectorize
# Workers AI
# 사이트맵(https://developers.cloudflare.com/sitemap-0.xml)을 사용하여 각 제품에 대한 공식문서를 찾아보세요.
# 여러분이 제출한 내용은 다음 질문으로 테스트됩니다:
# "llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?"
# "Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?"
# "벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?"
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 Streamlit app과 함께 깃허브 리포지토리에 링크를 넣습니다.

#https://share.streamlit.io/

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader,SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st

def get_llm(openai_api_key):
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini", # 가성비 모델 추천
        openai_api_key=openai_api_key
    )
    return llm

answers_prompt = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}

""")

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | get_llm(openai_api_key)
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "question":question,
    #         "context":doc.page_content
    #     })
    #answers.append(result.content)
    return {
        "question":question,
        "answers":[
            {
                "answer":answers_chain.invoke(
                    {"question": question,"context":doc.page_content}
                ).content ,
                "source": doc.metadata["source"] ,
                "date": doc.metadata["lastmod"], 
            } 
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | get_llm(openai_api_key)
    condensed ="\n\n".join(f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers)
    # for answer in answers:
    #     condensed += f"Answer:{answer['answer']}\Source:{answer['source']}\Date:{answer['date']}\n"
    #st.write(condensed)
    return choose_chain.invoke({
            "question":question,
            "answers": condensed,
        })

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
        # text = header.get_text()
        # return text
    if footer:
        footer.decompose()
    return (
            str(soup.get_text())
            # .replace("\n"," ")
            # .replace("nExplore"," ")
            )

@st.cache_data(show_spinner="Loading website..")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(url,
                            filter_urls=[r"^(.*\/ai-gateway\/).*", r"^(.*\/vectorize\/).*",],
                           parsing_function=parse_page)
    loader.requests_per_second = 1  # 차단당하지 않도록 1초단위로 요청시간 설정 
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs,OpenAIEmbeddings())
    #st.write(docs)   #Fetching pages 문구가 터미널 콘솔에 보임 .
    return vector_store.as_retriever() 

st.set_page_config(
    page_title="Site GPT",
    page_icon="👩🏻‍💻",
)

html2text_transformer = Html2TextTransformer()

st.markdown("""
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
""")

with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key",type="password")
    url = st.text_input("Write down a url",placeholder="https://example.com")

if not openai_api_key:
    st.error("Pls input open ai api key .")
elif url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Pls write down a sitemap url . ")
            #https://developers.cloudflare.com/sitemap-0.xml

    else:
        retriever = load_website(url)
        # docs = retriever.invoke("What is the price of Gemini 3?")
        # docs
        query = st.text_input("Ask a question to the website")
        if query:
            chain = {"docs":retriever,"question":RunnablePassthrough()} | RunnableLambda(get_answers)| RunnableLambda(choose_answer)
            result = chain.invoke(query)
            st.write(result.content.replace("$","\$"))
