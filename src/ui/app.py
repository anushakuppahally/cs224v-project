import datetime
from pathlib import Path
import numpy as np
import json
import h5py
from typing import List, Dict, Any
from together import Together
import streamlit as st
import faiss
from datasets import load_dataset
from src.qa.system import ElectionQASystem
# Streamlit UI code (src/ui/app.py):
def create_streamlit_app():
    st.title("2020 Election Q&A System")
    
    # Initialize QA system
    qa_system = ElectionQASystem(
        embeddings_dir="data/processed/embeddings",
        articles_file="data/raw/articles.json"
    )
    
    # Language selection
    language = st.selectbox(
        "Select Language",
        options=["en", "es"],
        format_func=lambda x: "English" if x == "en" else "Spanish"
    )
    
    # Query input
    query = st.text_input("Enter your question about the 2020 US Election:")
    
    if query:
        with st.spinner("Searching for relevant information..."):
            # Get relevant context
            context = qa_system.get_relevant_context(query, language)
            
            # Generate answer
            answer = qa_system.generate_answer(query, context, language)
            
            # Display results
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Sources:")
            for article in context:
                with st.expander(f"Source: {article['source']} - {article['date']}"):
                    st.write(f"**{article['title']}**")
                    st.write(article['text'][:500] + "...")