import time 
import streamlit as st


#터미널에서 streamlit run Home.py 실행 

st.title("Document GPT")

# with st.chat_message("human"):
#     st.write("Hello")

# with st.chat_message("ai"):
#     st.write("How are you ?")


# with st.status("Embeddings ... ",expanded=True) as status:
#     time.sleep(3)
#     st.write("Getting the file")
#     time.sleep(3)
#     st.write("Embedding the file ")
#     time.sleep(3)
#     st.write("Caching the file")
#     status.update(label="Error",state="error")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

def send_message(message,role,save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message":message, "role":role})
   

#메시지 이력을 다시 출력함(repainting) 그러나 세션스테이트에 저장은 하지 않음 
#    
for message in st.session_state["messages"]:
    send_message(message["message"],message["role"],save=False)


message = st.chat_input("send a message to ai")

if message:
    send_message(message,"human")
    time.sleep(2)
    send_message(f"you said: {message}","ai")

    with st.sidebar:
        st.write(st.session_state)