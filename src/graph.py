
import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    score: float


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536
)

vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def retrieve(state: AgentState):
    results = vector_store.similarity_search_with_score(
        state["question"],
        k=3
    )

    context_parts = []

    for document, score in results:
        page = document.metadata.get("page", "Unknown")

        if isinstance(page, (int, float)):
            page += 1

        context_parts.append(
            f"Page {page}:\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    average_score = (
        sum(score for _, score in results) / len(results)
        if results else 0.0
    )

    return {
        "context": context,
        "score": average_score
    }


def generate(state: AgentState):
    prompt = f"""
Answer the question using only the provided context
from the Agentic AI eBook.

If the answer is not available in the context, respond with:
"I couldn't find the answer in the Agentic AI eBook."

Context:
{state["context"]}

Question:
{state["question"]}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()