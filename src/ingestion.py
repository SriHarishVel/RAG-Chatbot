
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/Ebook-Agentic-AI.pdf"


def load_and_split_pdf():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    print(f"Total pages loaded: {len(pages)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(pages)

    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0].page_content[:500])

        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)

    return chunks


if __name__ == "__main__":
    chunks = load_and_split_pdf()