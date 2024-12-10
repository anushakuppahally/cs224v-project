import datetime
from pathlib import Path
import numpy as np
import json
import h5py
import torch
import gc
from typing import List, Dict, Any
from together import Together
import streamlit as st
from tqdm import tqdm
import faiss
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
import pickle


def load_election_dataset():
    """Load and instantiate dataset of articles """
    # dataset = load_dataset(
    #     "stanford-oval/ccnews", 
    #     name="2020",
    #     split="train", 
    #     streaming=True
    # ).filter(lambda article: article["language"] in ["en", "es"])
    # return dataset
    with open('data/raw/articles.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def filter_articles(article):
    """Filter articles to keep only election-related content from 2020"""
    # check if has needed fields 
    required_fields = ["plain_text", "published_date", "language"]
    if not all(field in article for field in required_fields):
        return False
    # date filters 
    start_date = datetime.datetime(2020, 11, 3, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2020, 11, 10, tzinfo=datetime.timezone.utc)
    
    # filter for date and check formatting 
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
        
    # filter for language
    if article["language"] not in ["en", "es"]:
        return False
        
    # election related keywords
    keywords = [
        # English keywords
        "election", "vote", "voting", "Trump", "Biden", 
        "ballot", "electoral college", "swing state", "battleground state",
        "polls", "polling", "democrat", "republican", "presidential",
        "voter", "precinct", "mail-in", "absentee",
         "supreme court", "electoral",
        
        # Spanish keywords
        "elección", "voto", "votar", "boleta", "papeleta",
        "colegio electoral", "estado pendular", "encuestas", "demócrata",
        "republicano", "presidencial", "votante", "recinto", 
        "correo", "ausente", "corte suprema", "electoral",
        "urnas", "sufragio", "comicios", "escrutinio"
    ]
    text = article["plain_text"].lower()
    if not any(kw.lower() in text for kw in keywords):
        return False
        
    return True


@st.cache_data(show_spinner=False, persist="disk")
def process_articles(dataset):
    """Process articles into collections without translation"""
    # check if cached file exists
    cache_file = Path('data/processed/articles.json')
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        print("Loading cached processed articles...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

        
    print("Processing articles from scratch...")
    articles = {"en": [], "es": []}
    article_id = 0
    
    for article in tqdm(dataset, desc="Processing articles"):
        lang = article["language"]
        processed = {
            "id": article_id,
            "text": article["plain_text"],
            "title": article["title"],
            "date": article["published_date"],
            "source": article["sitename"],
            "url": article["requested_url"],
            "lang": lang
        }
        articles[lang].append(processed)
        article_id += 1
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessed {len(articles['en'])} English articles")
    print(f"Processed {len(articles['es'])} Spanish articles")
    
    return articles

@lru_cache(maxsize=1000)
def get_embedding_cached(text: str, client, model_api_string: str):
    """Cache embeddings to avoid re-computing for identical texts"""
    response = client.embeddings.create(
        input=[text],
        model=model_api_string
    )
    return response.data[0].embedding



def generate_embeddings(articles, cache_dir="embeddings_cache"):
    """Generate embeddings using SentenceTransformer"""
    # cache dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "embeddings_cache.pkl")
    
    # try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass

    # multilingual model for embeddings
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    embeddings = {"en": {"embeddings": [], "article_ids": []},
                 "es": {"embeddings": [], "article_ids": []}}
    
    batch_size = 8
    
    for lang in tqdm(["en", "es"], desc="Processing languages"):
        articles_lang = articles[lang]
        texts = [article["text"] for article in articles_lang]
        
        # generate embeddings
        batch_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        embeddings[lang]["embeddings"] = batch_embeddings
        embeddings[lang]["article_ids"] = np.array([art["id"] for art in articles[lang]], dtype=np.int32)
        
        gc.collect()
    
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
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
