import faiss
import os
import streamlit as st
from backend.pdf_processing import extract_text_from_pdf
from backend.embeddings import generate_embeddings, initialize_faiss, add_to_faiss
from backend.query_engine import query_index
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Set Page Configuration
st.set_page_config(
    page_title="Content Engine Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for UI Enhancements
st.markdown(
    """
    <style>
        body {
            background-color: #f7f7f7;
            color: #333;
        }
        .chat-box {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            text-align: right;
            background-color: #007bff;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            display: inline-block;
        }
        .bot-message {
            text-align: left;
            background-color: #e9ecef;
            color: #333;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            display: inline-block;
        }
        .stTextInput input {
            background-color: #ffffff;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load or Initialize FAISS Index
if os.path.exists("faiss_index.bin"):
    faiss_index = faiss.read_index("faiss_index.bin")
else:
    faiss_index = initialize_faiss(embedding_model.get_sentence_embedding_dimension())

# Main App
st.title("🤖 Content Engine Chatbot")
st.subheader("Your Document Intelligence Assistant")
st.write(
    """
    Use this chatbot to:
    - Upload and analyze your documents.
    - Query insights from multiple PDFs.
    - Compare content between documents.
    """
)

# File Upload Section
st.header("📂 Upload PDF Documents")
uploaded_files = st.file_uploader(
    "Drag and drop or click to upload PDFs", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.header("Extracted Content")
    for file in uploaded_files:
        content = extract_text_from_pdf(file)
        if not content.strip():
            st.warning(f"No text found in {file.name}. Skipping.")
            continue
        st.text_area(f"Content of {file.name}", content[:1000], height=200)  # Display first 1000 chars
        
        # Process embeddings
        embeddings = generate_embeddings(content)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)  # Ensure 2D shape
        faiss_index = add_to_faiss(faiss_index, embeddings)

    # Save FAISS index
    faiss.write_index(faiss_index, "faiss_index.bin")
    st.success("Documents processed and saved successfully!")

# Query Section
st.header("🔍 Query Your Documents")
user_query = st.text_input("Enter your question:")
if user_query:
    query_embedding = embedding_model.encode(user_query)
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)  # Ensure 2D shape

    # Corrected query_index function call
    distances, indices = query_index(faiss_index, query_embedding, embedding_model)

    st.write("Query Results:")
    results = []
    for idx in indices[0]:  # Ensure correct index handling
        matching_file = uploaded_files[idx]
        snippet = extract_text_from_pdf(matching_file)[:500]
        st.markdown(f"**Document:** {matching_file.name}")
        st.markdown(f"**Snippet:** {snippet}")
        results.append({"Document": matching_file.name, "Snippet": snippet})

    # Download query results as CSV
    if results:
        df_results = pd.DataFrame(results)
        st.download_button(
            label="Download Results as CSV",
            data=df_results.to_csv(index=False),
            file_name="query_results.csv",
            mime="text/csv"
        )

# Compare Documents Section
if len(uploaded_files) > 1:
    st.header("🛠️ Compare Documents")
    doc1 = st.selectbox("Select Document 1", [file.name for file in uploaded_files])
    doc2 = st.selectbox("Select Document 2", [file.name for file in uploaded_files])
    
    comparison_query = st.text_input("Enter your comparison query:")
    if comparison_query:
        content1 = extract_text_from_pdf([f for f in uploaded_files if f.name == doc1][0])
        content2 = extract_text_from_pdf([f for f in uploaded_files if f.name == doc2][0])
        
        embeddings1 = generate_embeddings(content1)
        embeddings2 = generate_embeddings(content2)
        
        similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
        st.write(f"**Similarity between {doc1} and {doc2}:** {similarity:.2f}")