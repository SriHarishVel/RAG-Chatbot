import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# Load environment variables
load_dotenv()


# Define the agent state
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    score: float


# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536
)


# Connect to Pinecone
vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)


# Initialize Groq chat model
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# Node 1: Retrieve relevant documents
def retrieve(state: AgentState):

    question = state["question"]

    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    # Combine retrieved chunks into context
    context_parts = []

    for document, score in results:
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"Page {page + 1 if isinstance(page, (int, float)) else page}:\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Average similarity score
    average_score = (
        sum(score for _, score in results) / len(results)
        if results else 0.0
    )

    return {
        "context": context,
        "score": average_score
    }


# Node 2: Generate an answer using Groq
def generate(state: AgentState):

    question = state["question"]
    context = state["context"]

    prompt = f"""
You are a helpful chatbot answering questions about Agentic AI.

Answer the user's question using ONLY the provided context
from the Agentic AI eBook.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain the answer, clearly say:
   "I couldn't find the answer in the Agentic AI eBook."
3. Do not invent facts.
4. Keep the answer clear and easy to understand.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# Build the LangGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()