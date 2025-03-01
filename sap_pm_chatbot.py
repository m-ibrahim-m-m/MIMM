import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
from nltk.corpus import stopwords  # Replace spaCy with NLTK for lighter keyword extraction

# Force CPU usage and disable gradients
torch.set_grad_enabled(False)
torch.set_default_tensor_type(torch.FloatTensor)

# Load lightweight models
@st.cache_resource
def load_models():
    # Load DistilBERT (smaller than BERT)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Load NLTK stopwords (lighter than spaCy)
    import nltk
    nltk.download("stopwords")
    return tokenizer, model, stopwords.words("english")

tokenizer, model, STOPWORDS = load_models()

# Precompute embeddings for predefined responses (saves runtime resources)
@st.cache_data
def precompute_embeddings():
    embeddings = {}
    for key in responses:
        inputs = tokenizer(key, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings[key] = embedding
    return embeddings

response_embeddings = precompute_embeddings()

# Predefined responses (unchanged)
responses = {
    "How do I create a maintenance order?": "To create a maintenance order...",
    # ... (other responses)
    "default": "I'm sorry, I don't understand..."
}

# Simplified keyword extraction (no spaCy)
def extract_keywords(text):
    words = text.lower().split()
    return " ".join([w for w in words if w not in STOPWORDS])

# Lightweight paraphrasing (removed T5; use simpler logic)
def paraphrase_query(text):
    return text.replace("how", "what is the process to").replace("do I", "can one")

# Optimized response generation
def get_response(user_input):
    user_input = user_input.lower()
    
    # Compute user embedding
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    user_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Compare with precomputed embeddings
    similarities = {
        key: cosine_similarity([user_embedding], [emb])[0][0]
        for key, emb in response_embeddings.items()
    }
    
    best_match = max(similarities, key=similarities.get)
    if similarities[best_match] > 0.95:  # Lowered threshold
        return responses[best_match]
    else:
        keywords = extract_keywords(user_input)
        paraphrased = paraphrase_query(user_input)
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(keywords + ' ' + paraphrased)}"
        return f"Try searching SAP Community: [Google]({search_url})"

# Streamlit app (unchanged UI)
def main():
    st.title("SAP Plant Maintenance Chatbot")
    st.write("Welcome! How can I assist you today?")
    
    # Sidebar and chat logic...
    # (Same as your original code)

if __name__ == "__main__":
    main()
