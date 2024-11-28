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

def create_streamlit_app():
    st.title("2020 Election Q&A System")
    
    # initialize QA system
    qa_system = ElectionQASystem(
        embeddings_dir="data/processed/embeddings",
        articles_file="data/processed/articles.json"
    )
    
    def get_bilingual_context(qa_system, query, k=4):
        # get context from both languages
        en_context = qa_system.get_relevant_context(query, "en", k=2)
        es_context = qa_system.get_relevant_context(query, "es", k=2)
        
        # combine contexts
        combined_context = en_context + es_context
        
        # sort by relevance 
        # and limit to k total results
        return combined_context[:k]
 
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.markdown("""
        <style>
        .element-container {
            word-wrap: break-word;
            max-width: 100%;
        }
        
        .stMarkdown {
            word-wrap: break-word;
            white-space: pre-wrap;
            overflow-wrap: break-word;
            max-width: 100%;
        }
        
        /* Make sure code blocks also wrap */
        code {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
        }
        </style>
    """, unsafe_allow_html=True)

    def display_chat_history():
        for message in st.session_state.chat_history:
            # user message
            st.write("💁🏽‍♀️ **User:**")
            st.markdown(f"<div style='word-wrap: break-word; max-width: 100%;'>{message['query']}</div>", unsafe_allow_html=True)
            st.markdown("---")
            
            # assistant message
            st.write("🤖 **Assistant:**")
            st.markdown(
                f"<div style='word-wrap: break-word; white-space: pre-wrap; max-width: 100%;'>{message['answer'].replace('*', '').strip()}</div>", 
                unsafe_allow_html=True
            )
            st.markdown("---")

    # display chat history before the input box

    display_chat_history()
    
    # query input using a unique key for each render
    query = st.text_input(
        "Enter your question about the 2020 US Election:",
        key=f"query_{len(st.session_state.chat_history)}"  # Unique key based on chat history length
    )
    
    if st.button("Submit") and query:
        with st.spinner("Searching for relevant information..."):
            context = get_bilingual_context(qa_system, query)
            if len(context) == 0:
                st.warning("No relevant articles found. The assistant will provide a general response.")
            # pass chat history to generate_answer
            else:
                with st.spinner("Generating answer..."):
                    answer = qa_system.generate_answer(
                        query, 
                        context, 
                        chat_history=st.session_state.chat_history
                    )
                
            st.session_state.chat_history.append({
                'query': query,
                'answer': answer
            })
        
        st.rerun()