import streamlit as st
from src.graph import app as rag_graph

# Page configuration
st.set_page_config(
    page_title="Agentic AI eBook Chatbot",
    layout="centered"
)

# Title
st.title("Agentic AI eBook Chatbot")

st.write(
    "Ask questions about the Agentic AI eBook. "
    "Answers are grounded in the uploaded document."
)

# Required model disclosure
st.info(
    "Model disclosure: This chatbot uses Gemini for document embeddings "
    "and Groq for answer generation. These are alternative models used "
    "in place of the OpenAI models specified in the assignment reference. "
    "The specified OpenAI free-tier access was not available, so "
    "alternative models with available free-tier access were used."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "score" in message:
            st.caption(
                f"Average retrieval similarity score: "
                f"{message['score']:.4f}"
            )

            with st.expander("View retrieved context"):
                st.text(message["context"])

# Chat input
question = st.chat_input("Ask a question about Agentic AI...")

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching the eBook and generating an answer..."):

            try:
                result = rag_graph.invoke({
                    "question": question,
                    "context": "",
                    "answer": "",
                    "score": 0.0
                })

                answer = result["answer"]
                score = result["score"]
                context = result["context"]

                st.markdown(answer)

                st.caption(
                    f"Average retrieval similarity score: {score:.4f}"
                )

                with st.expander("View retrieved context"):
                    st.text(context)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "score": score,
                    "context": context
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")