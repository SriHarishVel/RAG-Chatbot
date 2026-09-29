
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Step 1: Load the API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found. Check your .env file.")

# Step 2: Initialize the Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
    output_dimensionality=1536
)

# Step 3: Test the embedding model
text = "AI agents can use tools to complete tasks."

vector = embeddings.embed_query(text)

print("Vector length:", len(vector))
print("First 5 numbers:", vector[:5])