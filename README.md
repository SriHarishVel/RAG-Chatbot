# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers user questions using information retrieved from the provided Agentic AI eBook.

The project combines document ingestion, semantic search, vector storage, and a LangGraph workflow to generate context-grounded responses and handle questions outside the knowledge base.

## Features

- PDF document loading and text extraction
- Text chunking with configurable chunk size and overlap
- Semantic embeddings using Google Gemini
- Vector storage and similarity search using Pinecone
- LangGraph-based retrieval and answer-generation workflow
- Context-grounded answers using Groq
- Streamlit chat interface
- Displays retrieved context chunks and retrieval similarity scores
- Handles out-of-scope questions with a predefined fallback response

## Technology Stack

| Component              | Technology                           |
| ---------------------- | ------------------------------------ |
| Programming Language   | Python                               |
| RAG Framework          | LangChain                            |
| Workflow Orchestration | LangGraph                            |
| Embedding Model        | Google Gemini `gemini-embedding-001` |
| Vector Database        | Pinecone                             |
| LLM Provider           | Groq                                 |
| Chat Model             | `openai/gpt-oss-20b`                 |
| User Interface         | Streamlit                            |
| PDF Processing         | PyPDF                                |

### Model Disclosure

This project uses Google Gemini for embeddings and Groq for answer generation as alternatives to the OpenAI models mentioned in the assignment reference.

The application uses the models and services listed above for its RAG pipeline.

## Project Structure

```text
rag-agentic-aigit status/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── app.py
│   ├── chat_test.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── test_embeddings.py
│   └── upload_to_pinecone.py
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.12 or compatible Python version
- Google Gemini API key
- Groq API key
- Pinecone API key
- Pinecone index configured for 1536-dimensional dense vectors with cosine similarity

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag-agentic-ai
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your public GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a template.

Add the following configuration:

```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

Replace the placeholder values with your own API keys.

**Security:** Never commit your actual `.env` file or expose API keys in source code, screenshots, or public repositories.

## Document Ingestion and Indexing

The provided Agentic AI eBook is processed into text chunks and indexed in Pinecone for semantic retrieval.

Place the assignment-provided PDF in the `data/` directory.

Run the ingestion script:

```bash
python -m src.ingestion
```

Upload the generated document chunks and embeddings to Pinecone:

```bash
python -m src.upload_to_pinecone
```

The upload script stores the document chunks and their vector embeddings in the configured Pinecone index.

**Note:** Run ingestion and indexing before querying the chatbot if the Pinecone index has not already been populated.

## Running the Application

Start the Streamlit interface from the project root:

```bash
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal to interact with the chatbot.

The interface allows users to:

1. Enter a question about the Agentic AI eBook.
2. Receive an answer generated using retrieved document context.
3. Inspect the retrieved context chunks.
4. View the retrieval similarity score.

## Retrieval and Answer Generation

The application follows a Retrieval-Augmented Generation workflow:

1. The user's question is converted into an embedding using Gemini.
2. Pinecone performs semantic similarity search against the indexed eBook chunks.
3. The most relevant chunks are retrieved.
4. LangGraph passes the retrieved context and question to the generation step.
5. Groq generates an answer grounded in the retrieved context.
6. The application displays the answer, retrieved context, and similarity score.

The chatbot is designed to avoid answering questions that are not supported by the indexed eBook.

### Out-of-Scope Handling

When the retrieved context does not provide sufficient relevant information, the chatbot returns:

> I couldn't find the answer in the Agentic AI eBook.

This helps prevent the chatbot from responding to unrelated questions using unsupported information.

## Testing

Run the chatbot test script:

```bash
python -m src.chat_test
```

Run the retrieval test:

```bash
python -m src.retrieve_test
```

Run these commands from the project root with the virtual environment activated and the required environment variables configured.
