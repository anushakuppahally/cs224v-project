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

# Data loading and processing functions (keeping previous functions)
def load_election_dataset():
    """Load and initialize the CCNews dataset for 2020 election coverage"""
    # dataset = load_dataset(
    #     "stanford-oval/ccnews", 
    #     name="2020",
    #     split="train", 
    #     streaming=True
    # ).filter(lambda article: article["language"] in ["en", "es"])
    # return dataset
    with open('data/raw/articles.json', 'r') as file:
        data = json.load(file)
    return data

# [Previous functions remain the same...]
def filter_articles(article):
    """Filter articles to keep only election-related content from 2020"""
    # Check if article has required fields
    required_fields = ["plain_text", "published_date", "language"]
    if not all(field in article for field in required_fields):
        return False
    # Define filters
    start_date = datetime.datetime(2020, 11, 4, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2020, 11, 5, tzinfo=datetime.timezone.utc)
    # Check if date is within election period (Nov 4-5, 2020)
    
    # Parse date
    try:
        date = datetime.datetime.strptime(article["published_date"], "%Y-%m-%d")
        if date.year != 2020:
            return False
    except:
        return False
    
    article_date = datetime.datetime.strptime(
        article["published_date"], 
        "%Y-%m-%d"
    ).replace(tzinfo=datetime.timezone.utc)
    if not (start_date <= article_date <= end_date):
        return False
        
    # Check language
    if article["language"] not in ["en", "es"]:
        return False
        
    # Check for election-related keywords
    keywords = ["election", "vote", "voting", "Trump", "Biden", "campaign", 
               "elección", "voto", "votar", "campaña"]
    text = article["plain_text"].lower()
    if not any(kw.lower() in text for kw in keywords):
        return False
        
    return True

def process_articles(dataset):
    """Process filtered articles into language-specific collections"""
    articles = {"en": [], "es": []}
    article_id = 0
    
    for article in dataset:
        processed = {
            "id": article_id,
            "text": article["plain_text"],
            "date": article["published_date"],
            "source": article["sitename"],
            "title": article["title"],
            "url": article["requested_url"]
        }
        articles[article["language"]].append(processed)
        article_id += 1
    # print(articles)
    with open('data/processed/articles.json', 'w') as f:
        json.dump(articles, f, indent=2)
    return articles

def generate_embeddings(articles):
    """Generate embeddings from Together API.

    Args:
        articles: a dictionary with language keys and lists of article dictionaries as values.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    api_key = config["api_key"]

    client = Together(api_key=api_key)
    model_api_string = "togethercomputer/m2-bert-80M-8k-retrieval"

    embeddings = {"en": {"embeddings": [], "article_ids": []},
                 "es": {"embeddings": [], "article_ids": []}}
    # print(articles)
    for lang in ['en','es']:
        # Prepare texts for embedding
        # print(lang)
        articles_lang = articles[lang]
        # print(articles_lang)
        texts = [article["text"] for article in articles_lang]
        # print(texts)
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=model_api_string
            )
            batch_embeddings = [emb.embedding for emb in response.data]
            all_embeddings.extend(batch_embeddings)
            
        embeddings[lang]["embeddings"] = np.array(all_embeddings)
        embeddings[lang]["article_ids"] = [art["id"] for art in articles[lang]]
    # print(embeddings)
    return embeddings
    
def save_embeddings(embeddings, save_dir):
    """Save embeddings and metadata to disk"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for lang in embeddings:
        with h5py.File(save_dir / f"{lang}_embeddings.h5", "w") as f:
            f.create_dataset("embeddings", data=embeddings[lang]["embeddings"])
            f.create_dataset("article_ids", data=embeddings[lang]["article_ids"])

def load_embeddings(load_dir):
    """Load embeddings and metadata from disk"""
    load_dir = Path(load_dir)
    embeddings = {}
    
    for lang in ["en", "es"]:
        with h5py.File(load_dir / f"{lang}_embeddings.h5", "r") as f:
            embeddings[lang] = {
                "embeddings": f["embeddings"][:],
                "article_ids": f["article_ids"][:]
            }
            
    return embeddings
