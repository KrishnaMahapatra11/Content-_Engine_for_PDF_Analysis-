import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_and_split_documents(pdf_folder):
    """Load text from PDFs and split them into smaller chunks."""
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            file_path = os.path.join(pdf_folder, filename)
            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)

            # Split the text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
            documents.extend(chunks)
    return documents

def main():
    # Step 1: Load and split text from PDFs
    pdf_folder = "."  # Current folder
    documents = load_and_split_documents(pdf_folder)

    # Step 2: Generate embeddings for the documents
    print("Generating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 3: Store the embeddings in FAISS
    print("Storing embeddings in FAISS...")
    vector_store = FAISS.from_texts(documents, embedding=embedding_model)

    # Save the FAISS index to disk
    vector_store.save_local("faiss_index")
    print("FAISS index saved successfully!")

    # Example query
    print("Querying the vector store...")
    query = "What are the key risks mentioned in Tesla's report?"
    results = vector_store.similarity_search(query, k=3)
    for idx, result in enumerate(results, start=1):
        print(f"Result {idx}: {result}")

if __name__ == "__main__":
    main()
