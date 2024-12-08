# ElectionBot: A Bilingual RAG-Based Question Answering System 🗳

An agent that helps analyze and output insights based on news sources, using TogetherAI and Streamlit. This project is meant to help analyze and summarize viewpoints about the 2020 Presidential Election from articles in both English and Spanish. 

## Overview 

This project helps implement a QA LLM system that helps users with questions about the election. Specifically, this system is able to help with cross-lingual analysis. 

## Features 

#### RAG System
- grounds the system's answers using article text
- identifies the two most relevant articles for each language 
- uses FAISS indexing to efficiently identify relevant articles 

#### Query Classification
- determines if a query is relevant before the system retrieves articles 
- when applicable, uses the past two interactions in chat history to determine this

#### User Interface 
- easy to use Streamlit UI 
- chat history and quick, concise responses

#### Feedback Logging 
- LLM based evaluation 
- creates logs for each day of usage


## Installation 

1. Obtain a dataset of your choosing of news articles. We retrieved articles based on work done here: https://huggingface.co/datasets/stanford-oval/ccnews. 

2. Clone the repository 

3. Install dependencies using a virtual environment 

```
pip install -r requirements.txt 
```

4. Obtain a Together API key in a config.json file 

5. Adjust filtering and prompt based on your use case 

6. Run the Streamlit app:

```
streamlit run run.py 
```
