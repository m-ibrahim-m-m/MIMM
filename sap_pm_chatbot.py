import streamlit as st
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import spacy

# Force CPU usage
torch.set_default_tensor_type(torch.FloatTensor)

# Load the transformer model and tokenizer
@st.cache_resource
def load_model():
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# Load spaCy for keyword extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load a paraphrasing model
paraphrase_pipeline = pipeline("text2text-generation", model="t5-small", device="cpu")

# Predefined responses for SAP Plant Maintenance
responses = {
    "How do I create a maintenance order?": "To create a maintenance order, go to transaction code 'IW31' in SAP. Fill in the required details such as equipment, description, planner group, activity type and priority, then save the order.",
    "How do I check equipment history?": "To check equipment history, use transaction code 'IE03'. Enter the equipment number and navigate to the 'History' tab.",
    "How do I schedule a maintenance plan?": "To schedule a maintenance plan, use transaction code 'IP10'. Enter the maintenance plan number and set the scheduling parameters.",
    "How do I report a breakdown?": "To report a breakdown, use transaction code 'IW51'. Enter the details of the breakdown, including the equipment and fault description.",
    "How do I view notifications?": "To view notifications, use transaction code 'IW29'. You can filter notifications by status, equipment, or date.",
    "default": "I'm sorry, I don't understand your query. Please try searching the SAP Help Portal for more information."
}

# Function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Function to extract keywords using spaCy
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(keywords)

# Function to paraphrase the query using T5
def paraphrase_query(text):
    paraphrased = paraphrase_pipeline(f"paraphrase: {text}", max_length=50, num_return_sequences=1)
    return paraphrased[0]["generated_text"]

# Function to get the chatbot's response
def get_response(user_input):
    user_input = user_input.lower()
    user_embedding = get_embedding(user_input)

    # Calculate similarity scores between user input and predefined responses
    similarities = {}
    for key, response in responses.items():
        response_embedding = get_embedding(key)
        similarity = cosine_similarity([user_embedding], [response_embedding])[0][0]
        similarities[key] = similarity

    # Find the most similar response
    best_match = max(similarities, key=similarities.get)
    if similarities[best_match] > 0.99:  # Threshold for similarity
        return responses[best_match]
    else:
        # Refine the query using advanced NLP
        keywords = extract_keywords(user_input)
        paraphrased_query = paraphrase_query(user_input)
        refined_query = f"{keywords} {paraphrased_query}"

        # Redirect to SAP Help Portal with the refined query
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(refined_query)}"
        return f"I couldn't find a specific answer. Please check the google for more information: [google or try SAP Community]({search_url})"

# Streamlit app
def main():
    st.title("SAP Plant Maintenance Chatbot")
    st.write("Welcome to the SAP Plant Maintenance Support Chatbot! How can I assist you today (Please select one of support options or press the hyperlink to use your words in Google search)?")

    # Display support options or hints
    st.sidebar.title("Support Options")
    st.sidebar.write("Here are some common queries you can ask:")
    st.sidebar.write("- How do I create a maintenance order?")
    st.sidebar.write("- How do I check equipment history?")
    st.sidebar.write("- How do I schedule a maintenance plan?")
    st.sidebar.write("- How do I report a breakdown?")
    st.sidebar.write("- How do I view notifications?")
    st.sidebar.write("Select exact query from the above options or type your query ended with 'in SAP' and press Enter.")

    # Initialize session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("You: ", key="user_input")

    # Chatbot response
    if user_input:
        with st.spinner("Processing your query..."):
            response = get_response(user_input)
            st.session_state.chat_history.append(f"You: {user_input}")
            st.session_state.chat_history.append(f"Chatbot: {response}")

    # Display chat history
    st.write("### Chat History")
    for message in st.session_state.chat_history:
        st.write(message)

    # Clear chat history button
    if st.button("Clear Chat History"):
        # Reset the chat history
        st.session_state.chat_history = []
        st.rerun()  # Force a rerun to update the UI immediately

if __name__ == "__main__":
    main()
