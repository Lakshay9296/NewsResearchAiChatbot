# News Research AI Chatbot

An AI-powered chatbot designed to help users research and answer questions based on news articles from provided URLs. It processes articles using natural language processing (NLP) and provides relevant answers based on the content. This project is categorized as a **Generative AI** project, leveraging state-of-the-art models for natural language understanding.

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Technologies](#technologies)
- [Project Structure](#project-structure)

## Overview

The **News Research AI Chatbot** allows users to input URLs containing news articles, which are processed and split into chunks using the Recursive Text Splitter. The data is embedded using Sentence Transformers and stored in a FAISS vector database. This enables efficient searching and answering of questions related to the article content.

## Demo

You can try out the chatbot live on the demo website: [News Research AI Chatbot Demo](https://news-research-chat.streamlit.app/)

## Features

- Process news articles from URLs.
- Use AI to generate embeddings for efficient search.
- Answer user questions based on the article's content.
- Utilize FAISS for quick and accurate retrieval of relevant information.
- Handle large articles by splitting them into manageable chunks.

## Technologies

- **Programming Language**: Python
- **Web Scraping**: Beautiful Soup and Requests modules for fetching and parsing article content from URLs.
- **AI/Embeddings**: Sentence Transformers for generating embeddings from article content.
- **LLM**: LLaMA 3 70B open-source model provided by Groq Cloud for natural language processing tasks.
- **Text Processing**: Recursive Text Splitter for dividing articles into smaller chunks.
- **Database**: FAISS Vector Database for storing and retrieving embeddings efficiently.

## Project Structure
```plaintext
.
├── app.py           # Main Streamlit application
├── requirements.txt  # List of project dependencies
└── README.md        # Project documentation
