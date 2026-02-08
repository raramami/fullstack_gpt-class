# (KR)
# QuizGPT를 구현하되 다음 기능을 추가합니다:
# 함수 호출을 사용합니다.
# 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
# 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
# 만점이면 st.ballons를 사용합니다.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
# st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.


import json 
from pathlib import Path
from re import split
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from networkx import prominent_group
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser

def reset_quiz():
    st.session_state["quiz_data"] = None

st.set_page_config(
    page_title="Quiz GPT",
    page_icon="🔮",
)

if "quiz_data" not in st.session_state:
    st.session_state["quiz_data"] = None

function = {
    "name":"create_quiz",
    "description":"function that create a list of question and answers and return a quiz ",
    "parameters": {
        "type":"object",
        "properties":{
            "questions":{
                "type":"array",
                "items":{
                    "type":"object",
                    "properties":{
                        "question":{
                            "type":"string"
                        },
                        "answers":{
                            "type":"array",
                            "items":{
                                "type":"object",
                                "properties":{
                                    "answer":{
                                        "type":"string",
                                    },
                                    "correct":{
                                        "type":"boolean"
                                    },
                                },
                                "required":["answer","correct"]
                            },
                        },
                    },
                    "required":["question","answers"]
                },
            },
        },
        "required":["questions"],
    },
}


st.title("Quiz GPT")

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading the file ...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    Path("./.cache/files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(  
        separator="\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia....")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=1)
    docs = retriever.get_relevant_documents(term)
    return docs
      

def get_quiz(_docs, _difficulty, _api_key):
    llm = ChatOpenAI(
        model="gpt-4o-mini", # 가성비 모델 추천
        temperature=0.1,
        openai_api_key=_api_key
    ).bind(functions=[function], function_call={"name": "create_quiz"})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a teacher. Create a {difficulty} quiz based on the context. Only use the provided context."),
        ("human", "Context: {context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": _docs})
    
    # 함수 호출 결과 파싱
    arguments = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(arguments)

if "quiz_data" not in st.session_state:
    st.session_state["quiz_data"] = None

def reset_quiz():
    st.session_state["quiz_data"] = None

with st.sidebar:
    docs = None
    openai_api_key = st.text_input("Input your OpenAI API Key",type="password")

    st.markdown("---")
    difficulty = st.select_slider("Select Difficulty", options=["Easy", "Normal", "Hard"], value="Normal",on_change=reset_quiz)

    choice = st.selectbox("Select what you want to use",(
        "File","Wikipedia"
    ),on_change=reset_quiz)

    if choice == "File" :
        file = st.file_uploader("Pls upload your file ",type=["pdf","txt","ppt"],on_change=reset_quiz)
        if file:
            docs = split_file(file)
            st.write(docs)

    else :
        topic = st.text_input("Search Wikipedia",on_change=reset_quiz)
        if topic : 
            docs = wiki_search(topic)


# --- 5. 메인 화면 로직 ---
if not openai_api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar.")
elif not docs:
    st.info("Upload a file or search Wikipedia to start!")
else:
    # 퀴즈가 없거나 새로 만들어야 할 때만 생성
    if st.session_state["quiz_data"] is None:
        with st.spinner("Generating quiz..."):
            st.session_state["quiz_data"] = get_quiz(docs, difficulty, openai_api_key)

    # 퀴즈 출력
    with st.form("quiz_form"):
        questions = st.session_state["quiz_data"]["questions"]
        user_answers = []
        
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            options = [a['answer'] for a in q['answers']]
            choice = st.radio(f"Select option for Q{i+1}", options, index=None, key=f"q_{i}", label_visibility="collapsed")
            user_answers.append(choice)
        
        submitted = st.form_submit_button("Submit Answers")
        
    if submitted:
        correct_count = 0
        for i, q in enumerate(questions):
            # 정답 찾기
            correct_answer = next(a['answer'] for a in q['answers'] if a['correct'])
            if user_answers[i] == correct_answer:
                correct_count += 1
                st.success(f"Q{i+1}: Correct!")
            else:
                st.error(f"Q{i+1}: Wrong! (Answer: {correct_answer})")
        
        # 결과 처리
        score = (correct_count / len(questions)) * 100
        st.subheader(f"Your Score: {score:.1f}%")
        
        if score == 100:
            st.balloons()
            st.success("Perfect! You are a master.")
        else:
            st.warning("Not perfect yet. Try again?")
            st.button("Reset and Retry", on_click=reset_quiz)