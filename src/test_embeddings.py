import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
    output_dimensionality=1536
)

text = "AI agents can use tools to complete tasks."

vector = embeddings.embed_query(text)

print("Vector length:", len(vector))
print("First 5 numbers:", vector[:5])