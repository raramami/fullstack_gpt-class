import streamlit as st
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="Fullstack GPT Home",
    page_icon="🤖",
)

st.title("Fullstack GPT")

with st.sidebar:
    st.sidebar.title("Sidebar Title")
    st.sidebar.text_input("XXX")

tab_one, tab_two , tab_thr = st.tabs(["A","B","C"])

with tab_one:
    st.write('a')

with tab_two:
    st.write('b')

with tab_thr:
    st.write('c')

st.subheader("Welcome to Streamlit")

st.markdown("""
    ### I love it . 
""")

# st.write("Hello")
# st.write([1,2,3,4])
# st.write({"key":1})

# st.write(PromptTemplate)
# p = PromptTemplate.from_template("xxx")
#p  #매직 !, 변수만 적어도 st.write() 와 동일한 효과 


model = st.selectbox("Choose your option",("GPT-3","GTP-4"))

if model == "GPT-3":
    st.write("cheap")
else:
    st.write("not cheap")
    value = st.slider("temperature",min_value=0.1, max_value=1.0,)
    st.write(value)