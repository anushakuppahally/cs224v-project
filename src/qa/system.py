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
import re
import faiss
import csv
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import pandas as pd 
from typing import List, Dict, Any
from dataclasses import dataclass
from src.data_processing.loader import load_embeddings

@dataclass
class EvaluationResult:
    #Helpfulness Relevance Accuracy Depth Creativity Level of Detail

    helpfulness_score: float
    relevance_score: float
    accuracy_score: float
    depth_score: float
    creativity_score: float
    level_of_detail_score: float
    feedback: str

with open("config.json", "r") as f:
    config = json.load(f)

class ElectionQASystem:
    def __init__(self, embeddings_dir: str, articles_file: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.articles_file = Path(articles_file)
        self.api_key = config["api_key"]
        # self.embedding_model = "togethercomputer/m2-bert-80M-8k-retrieval"
        self.together_client = Together(api_key = self.api_key)
        
        # initialize multilingual model
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.load_data()

        # logging eval 
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.eval_file = self.logs_dir / f"qa_evaluations_{datetime.now().strftime('%Y%m%d')}.csv"
        
        if not self.eval_file.exists():
            with open(self.eval_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'query',
                    'answer',
                    'relevance_score',
                    'accuracy_score',
                    'helpfulness_score',
                    'depth_score',
                    'creativity_score',
                    'level_of_detail_score',
                    'feedback'
                ])
    def load_data(self):
        """Load embeddings and articles"""
        # load articles
        with open(self.articles_file, encoding='utf-8') as f:
            self.articles = json.load(f)
            
        # load embeddings and create FAISS indices
        self.indices = {}
        embeddings = load_embeddings(self.embeddings_dir)
        
        for lang in embeddings:
            # index = faiss.IndexFlatL2(embeddings[lang]["embeddings"].shape[1]) 
            # check if embeddings exist and have data
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
            # get the last 2 exchanges for context
            recent_context = chat_history[-2:]  
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
    def evaluate_response(self, query: str, answer: str, context: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate the quality of the generated answer"""
        
        # reate evaluation prompt
        eval_prompt = f"""Evaluate this Q&A interaction about the 2020 US Presidential Election:

        Question: {query}
        Answer: {answer}

        Rate each aspect from 1-5 (single digit between 1 and 5)and provide brief feedback:
        - Helpfulness: How well did it address the user's needs?
        - Relevance: How well did it stay on topic?
        - Accuracy: How factually correct was it based on the context?
        - Depth: How thorough was the explanation?
        - Creativity: How well did it synthesize information?
        - Level of Detail: How specific and informative was it?

        Format your response as:
        Helpfulness: [score] - [feedback]
        Relevance: [score] - [feedback]
        Accuracy: [score] - [feedback]
        Depth: [score] - [feedback]
        Creativity: [score] - [feedback]
        Level of Detail: [score] - [feedback]"""

        # use LLM eval
        response = self.together_client.completions.create(
            prompt=eval_prompt,
            max_tokens=512,
            temperature=0.1, 
            top_p=0.3,       
            frequency_penalty=0.0, 
            presence_penalty=0.0,  
            model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
        )
        print(response.choices[0].text)
        feedback = response.choices[0].text.strip()
        
        # get scores from feedback
        scores = {
        'helpfulness': 0.0,
        'relevance': 0.0,
        'accuracy': 0.0,
        'depth': 0.0,
        'creativity': 0.0,
        'level of detail': 0.0
        }
    
        for line in feedback.split('\n'):
            line = line.strip()
            if ':' in line:
                try:
                    metric, content = line.split(':', 1)
                    metric = metric.lower().strip()
                    
                    # Extract score more robustly
                    if '-' in content:
                        score_part = content.split('-')[0].strip()
                        score_match = re.search(r'[1-5]', score_part)
                        if score_match:
                            scores[metric] = float(score_match.group())
                    else:
                        # Try to find any number 1-5 in the content
                        score_match = re.search(r'[1-5]', content)
                        if score_match:
                            scores[metric] = float(score_match.group())
                            
                except Exception as e:
                    print(f"Error parsing line: {line}, Error: {e}")
                    continue

        return EvaluationResult(
            helpfulness_score=scores['helpfulness'],
            relevance_score=scores['relevance'],
            accuracy_score=scores['accuracy'],
            depth_score=scores['depth'],
            creativity_score=scores['creativity'],
            level_of_detail_score=scores['level of detail'],
            feedback=feedback
        )


    def log_evaluation(
        self, 
        query: str, 
        answer: str, 
        evaluation: EvaluationResult
    ) -> None:
        """Log the Q&A interaction and its evaluation metrics to CSV"""
        
        # different metrics
        feedback_dict = {}
        for line in evaluation.feedback.split('\n'):
            if line.startswith('Helpfulness:'):
                feedback_dict['helpfulness'] = line.split('-')[1].strip()
            elif line.startswith('Relevance:'):
                feedback_dict['relevance'] = line.split('-')[1].strip()
            elif line.startswith('Accuracy:'): 
                feedback_dict['accuracy'] = line.split('-')[1].strip()
            elif line.startswith('Depth:'):
                feedback_dict['depth'] = line.split('-')[1].strip()
            elif line.startswith('Creativity:'):
                feedback_dict['creativity'] = line.split('-')[1].strip()
            elif line.startswith('Level of Detail:'):
                feedback_dict['level_of_detail'] = line.split('-')[1].strip()

        # Append to CSV
        with open(self.eval_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                query,
                answer,
                evaluation.helpfulness_score,
                evaluation.relevance_score, 
                evaluation.accuracy_score,
                evaluation.depth_score,
                evaluation.creativity_score,
                evaluation.level_of_detail_score,
                evaluation.feedback
            ])

    def generate_answer(self, query: str, context: List[Dict[str, Any]], chat_history=None) -> str:
        """Generate answer using LLM"""
        if not self.classify_query(query, chat_history):
            return {
                "answer": "I apologize, but I can only answer questions related to the 2020 US Presidential Election.",
                "evaluation": None
            }
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
        answer = response.choices[0].text
        evaluation = self.evaluate_response(query, answer, context)
        
        # log results
        self.log_evaluation(query, answer, evaluation)
        
        return {
            "answer": answer,
            "evaluation": evaluation
        }