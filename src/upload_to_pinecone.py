import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

index_name = os.getenv("PINECONE_INDEX_NAME")

if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("Pinecone API key not found in .env")

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("Gemini API key not found in .env")

if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not found in .env")

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536
)

# Load PDF
loader = PyPDFLoader("data/Ebook-Agentic-AI.pdf")
documents = loader.load()

print("Pages loaded:", len(documents))

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Connect to existing Pinecone index
vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

# Upload in small batches
batch_size = 20
total_batches = (len(chunks) + batch_size - 1) // batch_size

for start in range(0, len(chunks), batch_size):
    batch_number = start // batch_size + 1

    batch = chunks[start:start + batch_size]

    # Fixed IDs prevent duplicate records when retrying
    ids = [
        f"ebook-chunk-{i}"
        for i in range(start, start + len(batch))
    ]

    print(
        f"\nUploading batch {batch_number}/{total_batches}"
    )

    vector_store.add_documents(
        documents=batch,
        ids=ids
    )

    print("Batch uploaded successfully.")

    # Pause between batches to respect Gemini's free-tier limit
    if start + batch_size < len(chunks):
        print("Waiting 65 seconds for rate limit...")
        time.sleep(65)

print("\nAll chunks uploaded successfully!")