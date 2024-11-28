# New RAG and QA System functions
import datetime
from pathlib import Path
import numpy as np
import json
import h5py
import torch
from typing import List, Dict, Any
from together import Together
from sentence_transformers import SentenceTransformer
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
        # Remove Together AI specific code
        self.api_key = config["api_key"]
        # self.embedding_model = "togethercomputer/m2-bert-80M-8k-retrieval"
        self.together_client = Together(api_key = self.api_key)
        
        # initialize multilingual model
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.load_data()
        
    def load_data(self):
        """Load embeddings and articles"""
        # Load articles
        with open(self.articles_file, encoding='utf-8') as f:
            self.articles = json.load(f)
            
        # Load embeddings and create FAISS indices
        self.indices = {}
        embeddings = load_embeddings(self.embeddings_dir)
        
        for lang in embeddings:
            # index = faiss.IndexFlatL2(embeddings[lang]["embeddings"].shape[1]) 
            # Check if embeddings exist and have data
            if len(embeddings[lang]["embeddings"]) == 0:
                raise ValueError(f"No embeddings found for language {lang}")
                
            embedding_dim = embeddings[lang]["embeddings"].shape[1]
            
            index = faiss.IndexFlatIP(embedding_dim)
            
            # normalize the vectors before adding them
            faiss.normalize_L2(embeddings[lang]["embeddings"])
            index.add(embeddings[lang]["embeddings"])
            
            self.indices[lang] = {
                "index": index,
                "article_ids": embeddings[lang]["article_ids"]
            }

    def classify_query(self, query: str, chat_history=None) -> bool:
        """
        Determines if a query is relevant to the 2020 US Presidential Election.
        Considers chat history for context in follow-up questions.
        """
        # use chat history if available
        if chat_history and len(chat_history) > 0:
            # get the last few exchanges for context
            recent_context = chat_history[-2:]  # last 2 exchanges
            context_text = "\n".join([
                f"Previous User Question: {exchange['query']}\n"
                f"Previous Answer: {exchange['answer']}\n"
                for exchange in recent_context
            ])
            
            classify_prompt = f"""Given the following conversation about the 2020 US Presidential Election, determine if the new query is a relevant follow-up question.
            
            Previous Conversation:
            {context_text}
            
            New Query: {query}
            
            Is this query relevant to the ongoing conversation about the 2020 US Presidential Election? Answer with 'YES' or 'NO'.
            Consider:
            1. Is it a follow-up to the previous discussion?
            2. Does it ask for clarification about previously discussed election topics?
            3. Is it related to any aspect mentioned in the previous answers?
            """
        else:
            # original prompt for first question
            classify_prompt = f"""Determine if the following query is related to the 2020 US Presidential Election.
            Respond with either 'YES' or 'NO'.
            
            Query: {query}
            
            Is this query about the 2020 US Presidential Election?"""
        
        response = self.together_client.completions.create(
            prompt=classify_prompt,
            model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            max_tokens=10,
            temperature=0.1
        )
        
        return "YES" in response.choices[0].text.upper()


    def get_relevant_context(self, query: str, lang: str = "en", k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant articles using RAG"""

        # use same multilingual embedding model 
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 norm
        )
        
        # get language index 
        if lang not in self.indices:
            raise ValueError(f"Language {lang} not found in indices")
            
        D, I = self.indices[lang]["index"].search(
            query_embedding, k
        )
        
        # get relevant articles
        relevant_articles = []
        for idx in I[0]:
            article_id = self.indices[lang]["article_ids"][idx]
            article = next(
                art for art in self.articles[lang]
                if art["id"] == article_id
            )
            relevant_articles.append(article)
            
        return relevant_articles
        
    def generate_answer(self, query: str, context: List[Dict[str, Any]], chat_history=None) -> str:
        """Generate answer using LLM"""
        if not self.classify_query(query, chat_history):
            return "I apologize, but I can only answer questions related to the 2020 US Presidential Election."
        
        # format context 
        context_text = "\n\n".join(
            f"Title: {art['title']}\nContent: {art['text']}" 
            for art in context
        )
        
        # create prompt
        prompt = f"""You are an expert analyst of the 2020 US Presidential Election
        
       
        Based on these sources:
        {context_text}.

        Answer this question: {query}.

        
        Important:
        - Use both English and Spanish sources to provide a complete perspective
        - Keep Spanish article titles in their original language
        - Only include information from the provided sources
        - Include in text citations when relevant
        - If you cannot answer based on these sources, say so clearly"""

        # get response from LLM
        response = self.together_client.completions.create(
            prompt=prompt,
            max_tokens=512,
            top_p = 0.7,
            frequency_penalty = 0.1,
            presence_penalty = 0.1,
            model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

        )
        return response.choices[0].text