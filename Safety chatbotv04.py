import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import urllib.parse
import difflib
from langdetect import detect

@st.cache_resource
def load_model():
    return SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Upload and process Excel
@st.cache_data
def load_excel(file):
    df = pd.read_excel(file)
    df = df.dropna(subset=['question', 'response_text'])
    df['embedding'] = df['question'].apply(lambda x: model.encode(x))
    return df

# Initialize app
st.set_page_config(page_title="Multilingual Safety Chatbot", page_icon="🛡️", layout="wide")
st.title("🛡️ Safety Instruction Chatbot")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("📄 Upload Excel File", type=["xlsx"])
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
    dark_mode = st.checkbox("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("""
        <style>
            .stApp { background-color: #1e1e1e; color: white; }
            .stChatMessage { background-color: #2e2e2e; }
            a { color: #4dabf7; }
        </style>
    """, unsafe_allow_html=True)

# Model load
model = load_model()

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Handle input
if prompt := st.chat_input("Type your safety question (in Arabic or English)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_file is None:
        error = "⚠️ Please upload an Excel file with questions and responses."
        st.session_state.messages.append({"role": "assistant", "content": error})
        st.chat_message("assistant").markdown(error)
    else:
        with st.spinner("Analyzing your question..."):
            try:
                df = load_excel(uploaded_file)
                user_embedding = model.encode(prompt)
                similarities = cosine_similarity([user_embedding], list(df['embedding']))[0]
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])

                if best_score > 0.7:
                    response_text = df.iloc[best_idx]['response_text']
                    video_url = df.iloc[best_idx].get('video_url', '')
                    lang = detect(response_text)
                    final_response = f"{response_text}\n\n🧠 Confidence: {best_score:.2f}"

                    if video_url and isinstance(video_url, str) and video_url.startswith("http"):
                        final_response += f"\n📺 [Watch Video]({video_url})"

                    with st.chat_message("assistant"):
                        if lang == "ar":
                            st.markdown("<div class='rtl'>" + final_response + "</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(final_response, unsafe_allow_html=True)
                        if video_url:
                            st.video(video_url)

                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                else:
                    # Try partial match fallback
                    close_matches = difflib.get_close_matches(prompt.lower(), df['question'].str.lower(), cutoff=0.4)
                    if close_matches:
                        suggestions = "\n".join(f"- {m}" for m in close_matches)
                        msg = f"🔍 Did you mean:\n{suggestions}"
                    else:
                        search_query = prompt + " safety procedures"
                        encoded_query = urllib.parse.quote_plus(search_query)
                        google_url = f"https://www.google.com/search?q={encoded_query}"
                        msg = f"I couldn't find a good match.\n🔎 <a href='{google_url}' target='_blank'>Search Google</a>\n📧 Contact safety team: safety@company.com"

                    with st.chat_message("assistant"):
                        st.markdown(msg, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

            except Exception as e:
                err_msg = f"⚠️ An error occurred: {str(e)}"
                with st.chat_message("assistant"):
                    st.markdown(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

# Add RTL support CSS
st.markdown("""
<style>
.rtl {
  direction: rtl;
  text-align: right;
  font-family: "Cairo", sans-serif;
}
</style>
""", unsafe_allow_html=True)
