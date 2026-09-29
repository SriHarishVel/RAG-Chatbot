
import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.ingestion import load_and_split_pdf
from src.embeddings import embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found in .env")


def upload_documents():
    print("Loading and splitting the PDF...")

    chunks = load_and_split_pdf()

    if not chunks:
        print("No document chunks found.")
        return

    print(f"Uploading {len(chunks)} chunks to Pinecone...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index = pc.Index(PINECONE_INDEX_NAME)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    vector_store.add_documents(chunks)

    print("Documents uploaded successfully!")


if __name__ == "__main__":
    upload_documents()