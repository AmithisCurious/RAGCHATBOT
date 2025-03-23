"""
Streamlit app for the Educational RAG System
- Allows uploading PDFs
- Processes PDFs into Q&A pairs
- Provides a chat interface for querying the RAG system
"""

import os
import streamlit as st
import time
from dotenv import load_dotenv, find_dotenv
import torch
load_dotenv(find_dotenv())
# Use a fixed API key
claude_api_key = os.getenv("claude_api_key")
# Import functions from our backend module
# Adjust the import path to match your project structure
from Backend.DB_BACKEND import (
    setup_claude_client,
    setup_chroma_db,
    process_pdf_to_qa,
    query_rag_system,
    setup_deepseek_model,
    CHROMA_PERSIST_DIR
)

# Page configuration
st.set_page_config(
    page_title="CS Educational Assistant",
    page_icon="📚",
    layout="wide"
)
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None

# Sidebar for PDF upload and configuration
st.sidebar.subheader("GPU Information")
if torch.cuda.is_available():
    st.sidebar.success(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    st.sidebar.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    st.sidebar.info(f"CUDA Version: {torch.version.cuda}")
else:
    st.sidebar.error("❌ No GPU detected")
with st.sidebar:
    st.title("CS Educational Assistant")
    st.write("Upload computer science textbooks to build a knowledge base for guided learning.")
    
    
    # File uploader
    uploaded_file = st.file_uploader("Upload PDF Textbook", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_upload.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process button
        if st.button("Process PDF"):
            with st.spinner("Processing PDF into Q&A pairs..."):
                try:
                    # Initialize Chroma client if not already done
                    if st.session_state.chroma_client is None:
                        st.session_state.chroma_client = setup_chroma_db()
                    
                    # Process the PDF
                    qa_pairs = process_pdf_to_qa(
                        "temp_upload.pdf",
                        claude_api_key,
                        st.session_state.chroma_client
                    )
                    
                    st.success(f"Successfully processed {len(qa_pairs)} Q&A pairs from {uploaded_file.name}")
                    
                    # Clean up
                    os.remove("temp_upload.pdf")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
    
    # Collection stats
    st.subheader("Knowledge Base Stats")
    if st.session_state.chroma_client is None:
        try:
            st.session_state.chroma_client = setup_chroma_db()
        except Exception as e:
            st.error(f"Error connecting to vector database: {e}")
    
    if st.session_state.chroma_client is not None:
        try:
            collection = st.session_state.chroma_client.get_collection("cs_educational_qa")
            st.write(f"Total Q&A pairs: {collection.count()}")
        except Exception as e:
            st.write("No Q&A pairs in database yet.")
    
    # Load model button to save memory when not actively chatting
    if st.button("Load DeepSeek Model"):
        with st.spinner("Loading DeepSeek model..."):
            try:
                st.session_state.model, st.session_state.tokenizer = setup_deepseek_model()
                st.success("DeepSeek model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")

# Main area for chatting
st.title("Computer Science Educational Assistant")
st.write("""
This assistant is designed to help computer science students learn concepts through guided discovery.
Instead of giving direct answers, it provides learning roadmaps with resources and hints.
""")

# Chat history display
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Query input
query = st.chat_input("Ask a computer science question...")

if query:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Check if Chroma client is initialized
        if st.session_state.chroma_client is None:
            try:
                st.session_state.chroma_client = setup_chroma_db()
            except Exception as e:
                st.error(f"Error connecting to vector database: {e}")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "Error: Could not connect to the knowledge base."
                })
                st.stop()
        
        # Check if model is loaded
        if st.session_state.model is None or st.session_state.tokenizer is None:
            try:
                st.session_state.model, st.session_state.tokenizer = setup_deepseek_model()
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "Error: Could not load the DeepSeek model. Please try loading it from the sidebar first."
                })
                st.stop()
        
        # Get response from RAG system
        try:
            full_response = query_rag_system(
                query,
                st.session_state.chroma_client,
                st.session_state.model,
                st.session_state.tokenizer
            )
            
            # Simulated typing effect
            response = ""
            for chunk in full_response.split():
                response += chunk + " "
                time.sleep(0.01)
                message_placeholder.markdown(response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response
            })
        
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            message_placeholder.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": error_msg
            })

# Reset chat button
if st.button("Reset Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()