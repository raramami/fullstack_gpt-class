from langchain.document_loaders import AsyncChromiumLoader,SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st



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
            .replace("\n"," ")
            .replace("nExplore"," ")
            )

@st.cache_data(show_spinner="Loading website..")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(url,
                           filter_urls=[r"^(.*\/science\/).*",],
                           parsing_function=parse_page)
    loader.requests_per_second = 1  # 차단당하지 않도록 1초단위로 요청시간 설정 
    docs = loader.load_and_split(text_splitter=splitter)
    #st.write(docs)   #Fetching pages 문구가 터미널 콘솔에 보임 .
    return docs 

st.set_page_config(
    page_title="Site GPT",
    page_icon="👩🏻‍💻",
)

html2text_transformer = Html2TextTransformer()

st.title("Site GPT")

with st.sidebar:
    url = st.text_input("Write down a url",placeholder="https://example.com")


if url:
    #async chromium loader : playwright install 명령어로 설치 
    # loader = AsyncChromiumLoader(url)
    # docs = loader.load()
    # transformed = html2text_transformer.transform_documents(docs)
    # st.write(docs)

    # https://openai.com/index/frontier-risk-and-preparedness/

    if ".xml" not in url:
        with st.sidebar:
            st.error("Pls write down a sitemap url . ")
            #https://deepmind.google/sitemap.xml

    else:
       docs = load_website(url)
       st.write(docs)