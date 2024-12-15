# Content Engine

Content Engine is a **Retrieval Augmented Generation (RAG)** system designed to analyze, compare, and highlight differences across multiple PDF documents. It combines advanced natural language processing (NLP) techniques with an interactive user interface to provide insightful document analysis. With the integration of machine learning models, vector stores, and conversational frameworks, the Content Engine facilitates efficient information retrieval and generation.

---

## Features

- **Upload and Process PDFs**: Seamlessly upload and process multiple PDF documents.
- **Document Comparison**: Analyze and compare documents to identify and highlight key differences.
- **Retrieval Augmented Generation (RAG)**: Utilize state-of-the-art RAG techniques for effective content retrieval and response generation.
- **Chat History**: Maintain chat context to enable seamless, conversational querying.
- **Interactive UI**: Streamlit-based interface for a smooth and intuitive user experience.

---

## Technologies Used

The Content Engine leverages cutting-edge tools and frameworks:

- **Streamlit**: To create a dynamic web interface for document analysis.
- **LangChain**: For implementing conversational retrieval chains and RAG workflows.
- **HuggingFace Embeddings**: To generate embeddings for document content.
- **LlamaCpp**: As the local language model for contextual insights.
- **FAISS**: Vector store for managing and querying document embeddings efficiently.
- **PyPDFLoader**: For loading and processing PDF files.
- **RecursiveCharacterTextSplitter**: To split long text into manageable chunks for embedding.
- **ConversationBufferMemory**: To preserve chat history for contextual interactions.

---

## Prerequisites

To set up and run the Content Engine, ensure you have the following:

- **Python**: Version 3.7 or higher
- **Dependencies**: Install the required Python libraries (listed below).

### Required Python Libraries:

```bash
pip install streamlit langchain sentence-transformers chromadb transformers pypdf2 faiss-cpu
```

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd content-engine
```

### 2. Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Download and Preprocess Data

Download your target PDF documents (e.g., "Alphabet Inc.", "Tesla Inc.", "Uber Technologies Inc.") and preprocess them as follows:

1. Use **PyPDFLoader** or **PyMuPDF** to extract the text from PDFs.
2. Save the extracted content for embedding generation.

### 4. Generate Embeddings

Use **Sentence Transformer** to generate embeddings for the document content:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(document_text)
```

### 5. Set Up Vector Store

Store the generated embeddings in **FAISS** or **ChromaDB**:

```python
import chromadb
vector_store = chromadb.create_client()
vector_store.add(documents=document_embeddings)
```

---

## Development Workflow

### 1. Backend Configuration

- **LangChain**: Implement conversational retrieval chains for analyzing and comparing documents.
- **Embedding Generation**: Use **Sentence Transformer** to generate embeddings from the parsed PDF text.
- **Vector Storage**: Persist embeddings using **FAISS** or **ChromaDB** for efficient retrieval.

### 2. Frontend Development

- **Streamlit Interface**: Create an interactive interface for uploading, querying, and visualizing differences between documents.
- **Chatbot Integration**: Enable conversational queries powered by the **LLM** backend.

### 3. Query Engine

- Configure retrieval tasks based on embeddings stored in the vector store.
- Enable contextual insights with the **LlamaCpp** local LLM.

---

## Running the Application

1. **Start the Streamlit App**:

```bash
streamlit run app.py
```

2. **Upload PDF Documents**:
   - Use the interface to upload the desired PDF files.

3. **Query the System**:
   - Ask questions or request comparisons between the uploaded documents.

4. **View Insights**:
   - The system will retrieve, analyze, and display key differences or generate responses based on the documents.

---

## Example Use Case

### Scenario:
You upload three annual reports from different companies (e.g., Alphabet Inc., Tesla Inc., and Uber Technologies Inc.) and ask the system:

- "What are the key financial highlights for each company?"
- "Compare the sustainability initiatives of Alphabet and Tesla."

The Content Engine will analyze the documents, retrieve relevant information, and generate an insightful response.

---

## Future Enhancements

- **Support for Additional File Formats**: Extend support to Word documents and other text formats.
- **Improved Visualization**: Add graphical insights like charts for comparison.
- **Enhanced Language Model**: Integrate larger LLMs for deeper contextual understanding.
- **Cloud Deployment**: Host the application on platforms like AWS, Azure, or Google Cloud for accessibility.

---

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of the changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Special thanks to the developers and communities of:

- **LangChain**
- **HuggingFace**
- **Streamlit**
- **FAISS**
- **Sentence Transformers**

