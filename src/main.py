import streamlit as st
from src import query_pinecone, format_context, generate_answer

st.title("UFC Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if question:
        matches = query_pinecone(question, 10)
        context = format_context(matches)
        answer = generate_answer(question, context)

        st.write("Answer:", answer)

        with st.expander("Chat History"):
            for chat in st.session_state.chat_history:
                st.write(chat)
