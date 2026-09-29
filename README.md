
# Agentic AI eBook Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using the Agentic AI eBook.

The application retrieves relevant document chunks from a Pinecone vector database and uses a language model to generate answers based on the retrieved content.

## Features

- PDF loading and text chunking
- Gemini-based document embeddings
- Pinecone vector database for similarity search
- LangGraph workflow for retrieval and answer generation
- Groq-powered answer generation
- Streamlit user interface
- Retrieved context and similarity score display
- Fallback response for questions not covered by the eBook

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| PDF Processing | PyPDF |
| Embeddings | Google Gemini |
| Vector Database | Pinecone |
| LLM Workflow | LangGraph |
| Answer Generation | Groq |
| Frontend | Streamlit |

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── app.py
│   ├── embeddings.py
│   ├── graph.py
│   ├── ingestion.py
│   └── upload_to_pinecone.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10 or later
- Gemini API key
- Groq API key
- Pinecone API key
- Pinecone index with 1536 dimensions and cosine similarity

## Installation

Clone the repository:

```bash
git clone https://github.com/SriHarishVel/RAG-Chatbot.git
cd RAG-Chatbot
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

Add the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

Replace the placeholder values with your own API keys.

Never upload your `.env` file or expose API keys publicly.

## Document Ingestion

Place the Agentic AI eBook at:

```text
data/Ebook-Agentic-AI.pdf
```

Run the ingestion script:

```bash
python -m src.ingestion
```

This loads the PDF and splits it into chunks for retrieval.

## Upload to Pinecone

To upload document chunks to Pinecone, run:

```bash
python -m src.upload_to_pinecone
```

The script generates embeddings and stores the document vectors in the configured Pinecone index.

Run this step when setting up a fresh index. Avoid repeated uploads to an already populated index.

## Run the Application

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Open the local URL displayed in the terminal to access the chatbot.

Ask questions related to the Agentic AI eBook and view the generated answers and retrieved context.