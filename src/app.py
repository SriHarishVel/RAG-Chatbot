
import streamlit as st
from src.graph import app as rag_graph

st.set_page_config(
    page_title="Agentic AI eBook Chatbot",
    layout="centered"
)

st.title("Agentic AI eBook Chatbot")

st.write(
    "Ask questions about the Agentic AI eBook."
)

st.info(
    "This prototype uses Gemini for embeddings and Groq "
    "for answer generation instead of the OpenAI models "
    "specified in the assignment reference."
)

question = st.chat_input("Ask a question...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                result = rag_graph.invoke({
                    "question": question,
                    "context": "",
                    "answer": "",
                    "score": 0.0
                })

                st.markdown(result["answer"])

                with st.expander("Retrieved Context"):
                    st.write(result["context"])

                st.caption(
                    f"Retrieval similarity score: "
                    f"{result['score']:.4f}"
                )

            except Exception:
                st.error("Unable to generate an answer.")