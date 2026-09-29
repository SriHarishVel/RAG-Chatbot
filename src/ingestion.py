from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load the PDF
pdf_path = "data/Ebook-Agentic-AI.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Total pages:", len(documents))

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Step 3: Display the first chunk
if chunks:
    print("\nFirst chunk:")
    print(chunks[0].page_content)

    print("\nChunk metadata:")
    print(chunks[0].metadata)
