import streamlit as st
from .tool import setup_model


def display_chat():
    if st.session_state.get("model") is None:
        st.error("No model selected. Please select a model first.")
        return
    
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("say something ..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            response = st.write_stream(st.session_state.model.stream(st.session_state.history))
        st.session_state.history.append({"role": "assistant", "content": response})

def layout():
    st.title("Chat")
    setup_model()
    display_chat()
