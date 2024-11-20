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
    

if __name__ == "__main__":
    # For data processing script
    dataset = load_election_dataset()
    
    # Process and save data
    processed_articles = []
    for article in dataset:
        if filter_articles(article):
            processed_articles.append(article)
    # save_processed_data(processed_articles, "data/raw/articles.json")
    # Generate and save embeddings

    processed_articles = process_articles(processed_articles)
    embeddings = generate_embeddings(processed_articles)
    save_embeddings(embeddings, "data/processed/embeddings")
    
    # For Streamlit app
    create_streamlit_app()



