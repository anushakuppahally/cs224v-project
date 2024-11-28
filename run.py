# Core data processing functions
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
from src.data_processing.loader import save_embeddings, load_embeddings, load_election_dataset
from src.data_processing.loader import process_articles, filter_articles, generate_embeddings
from src.ui.app import create_streamlit_app
import sys
import io
from contextlib import contextmanager

with open("config.json", "r") as f:
    config = json.load(f)
api_key = config["api_key"]

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = stdout

# initialize session state variables if they don't exist
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False
if 'processed_articles' not in st.session_state:
    st.session_state.processed_articles = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

def initialize_data():
    """Initialize data only once per session"""
    if not st.session_state.data_loaded:
        with suppress_stdout():
            st.session_state.processed_articles = load_and_process_data()
            st.session_state.data_loaded = True
    
    if not st.session_state.embeddings_generated:
        with suppress_stdout():
            st.session_state.embeddings = get_or_generate_embeddings(st.session_state.processed_articles)
            st.session_state.embeddings_generated = True
    
    return st.session_state.processed_articles, st.session_state.embeddings

# caching to speed up process 
@st.cache_data(show_spinner=False)  
def load_and_process_data():
    """Load and process data with caching"""
    # load the dataset from the jsonl file
    print("Loading dataset...")
    dataset = load_election_dataset()

    # filter the articles
    print("Processing articles...")
    processed_articles = []
    for article in dataset:
        if filter_articles(article):
            processed_articles.append(article)

    processed_articles = process_articles(processed_articles)
    return processed_articles

@st.cache_data(show_spinner=False)
def get_or_generate_embeddings(processed_articles):
    """Generate or load embeddings with caching"""
    print("Generating/loading embeddings...")
    embeddings = generate_embeddings(processed_articles)
    save_embeddings(embeddings, "data/processed/embeddings")
    return embeddings

if __name__ == "__main__":
    # initialize data once
    processed_articles, embeddings = initialize_data()
    
    # Launch UI
    with suppress_stdout():
        create_streamlit_app()



