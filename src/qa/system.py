# New RAG and QA System functions
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
from src.data_processing.loader import load_embeddings

with open("config.json", "r") as f:
    config = json.load(f)

class ElectionQASystem:
    def __init__(self, embeddings_dir: str, articles_file: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.articles_file = Path(articles_file)
        self.api_key = config["api_key"]
        self.embedding_model  = "togethercomputer/m2-bert-80M-8k-retrieval"
        self.together_client = Together(api_key = self.api_key)
        self.load_data()
        
    def load_data(self):
        """Load embeddings and articles"""
        # Load articles
        with open(self.articles_file) as f:
            self.articles = json.load(f)
            
        # Load embeddings and create FAISS indices
        self.indices = {}
        embeddings = load_embeddings(self.embeddings_dir)
        
        for lang in embeddings:
            # index = faiss.IndexFlatL2(embeddings[lang]["embeddings"].shape[1]) 
            # Check if embeddings exist and have data
            if len(embeddings[lang]["embeddings"]) == 0:
                raise ValueError(f"No embeddings found for language {lang}")
                
            # Get embedding dimension
            embedding_dim = embeddings[lang]["embeddings"].shape[1]
            
            # Create and populate FAISS index
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(embeddings[lang]["embeddings"])
            
            self.indices[lang] = {
                "index": index,
                "article_ids": embeddings[lang]["article_ids"]
            }
    
    def get_relevant_context(self, query: str, language: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant articles using RAG"""
        # Encode query
        
        response = self.together_client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        query_embedding = np.array([response.data[0].embedding])
        
        # Search in appropriate language index
        D, I = self.indices[language]["index"].search(
            np.array([query_embedding]), k
        )
        
        # Get relevant articles
        relevant_articles = []
        for idx in I[0]:
            article_id = self.indices[language]["article_ids"][idx]
            article = next(
                art for art in self.articles[language] 
                if art["id"] == article_id
            )
            relevant_articles.append(article)
            
        return relevant_articles
    
    def generate_answer(self, query: str, context: List[Dict[str, Any]], language: str) -> str:
        """Generate answer using LLM"""
        # Format context
        context_text = "\n\n".join(
            f"Title: {art['title']}\nContent: {art['text']}" 
            for art in context
        )
        
        # Create prompt
        prompt = f"""Based on the following articles about the 2020 US Election, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""

        # Generate response using Together API
        response = self.together_client.complete(
            prompt=prompt,
            model="togethercomputer/llama-2-70b-chat",
            max_tokens=500,
            temperature=0.7
        )
        
        return response.output.text