import streamlit as st
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import time
import requests
from bs4 import BeautifulSoup
import os
st.set_page_config(page_title="News Research AI Chat", page_icon="📰")

api_key = os.getenv("api_key")

def fetch_url_data(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise requests.exceptions.RequestException(f": {url}")
    soup = BeautifulSoup(response.text, 'html.parser')
    tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
    taglist = []
    for tag in tags:
        taglist.append(tag.get_text(strip=True))
    text = ' '.join(taglist)
    return text

st.title("News Research AI Chatbot 📰")
st.markdown("---")

st.sidebar.title("News Article URLs")

urls = []
for i in range(4):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

process = st.sidebar.button("Process URLs")

# Initialize session state for chunks and FAISS index
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = []
if 'index' not in st.session_state:
    st.session_state['index'] = None

if process:
    valid_urls = [url for url in urls if url]  # Filter out empty URLs

    if not valid_urls:
        st.error("Please enter at least one valid URL.")
    else:
        try:
            with st.spinner("Loading the data..."):
                data = [fetch_url_data(url) for url in valid_urls]
                text = ' '.join(data)
                time.sleep(1.5)

            with st.spinner("Splitting into chunks..."):
                r_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
                chunks = r_splitter.split_text(text)
                st.session_state['chunks'] = chunks
                time.sleep(1.5)

            with st.spinner("Embedding the chunks..."):
                encoder = SentenceTransformer("all-mpnet-base-v2")
                vectors = encoder.encode(chunks)
                time.sleep(0.5)

            with st.spinner("Storing in database..."):
                dim = vectors.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(vectors)
                st.session_state['index'] = index
                time.sleep(1.5)

        except requests.exceptions.RequestException as e:
            st.error(f"Access Denied by URL: {e}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if st.session_state['chunks']:
    question = st.text_input("Enter your question:")

    if question and st.session_state['index'] is not None:
        with st.spinner("Processing your question..."):
            # Query the database
            encoder = SentenceTransformer("all-mpnet-base-v2")
            question_embedding = encoder.encode(question)
            distances, I = st.session_state['index'].search(np.array([question_embedding]), k=10)
            results = [st.session_state['chunks'][i] for i in I[0]]

            # LLM
            context = "\n".join(results)
            prompt = f"Based on the following information:\n{context}\n\nQuestion: {question}"

            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Use the following context to answer the user's question. User will mostly ask you to read the news article snippets provided and answer on the basis of that. All responses should be in markdown syntax. Present in a good way. Don't make your responses too short.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                model="llama3-70b-8192",
            )
            response = chat_completion.choices[0].message.content

            st.markdown(response)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.markdown("This chatbot processes news articles from given URLs using AI to help answer user questions based on the content.")
st.sidebar.markdown("It leverages the following technologies:")
st.sidebar.markdown("- **BeautifulSoup**: For web scraping and extracting article text")
st.sidebar.markdown("- **SentenceTransformers**: To convert article chunks into embeddings")
st.sidebar.markdown("- **FAISS**: For efficient similarity search and retrieval of article segments")
st.sidebar.markdown("- **Groq Cloud (Llama 3)**: For generating AI-based answers from the content using a large language model")
st.sidebar.markdown("- **Streamlit**: For building the user interface")
st.sidebar.markdown("---")
st.sidebar.markdown("**Currently using model: Llama 3 (70B parameters)**")
st.sidebar.markdown("**API Key: Groq Cloud**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Made with ❤️ by Lakshay Kumar**")

