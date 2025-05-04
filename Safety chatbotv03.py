import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Safety Chatbot with Excel", page_icon="🛡️", layout="wide")
st.title("🛡️ Safety Instruction Chatbot")
st.markdown("Upload an Excel file with: **question**, **answer**, **topic**")

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

def rephrase_query(query):
    keywords = [
        "safety procedures", "emergency guide", "workplace protocol",
        "OSHA rules", "industrial safety", "site hazard control"
    ]
    return f"{query} {np.random.choice(keywords)}"

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if not all(col in df.columns for col in ['question', 'answer', 'topic']):
            st.error("❌ Excel must have columns: question, answer, topic")
        else:
            st.success("✅ File loaded. Ask away!")
            questions = df['question'].tolist()
            embeddings = model.encode(questions, show_progress_bar=False)

            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "🛡️ Welcome! Ask me about safety procedures or type 'menu' to see topics."}]

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)

            if prompt := st.chat_input("❓ Ask your safety question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Handle special commands
                if prompt.lower() == "menu":
                    reply = "🧾 Available Topics:\n- Fire safety\n- First aid\n- PPE usage\n- PPE Training\n- Chemical spills\n- Earthquake response"
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)

                elif prompt.lower() in ["exit", "quit", "bye"]:
                    reply = "👋 Stay safe! Contact safety@company.com for emergencies."
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)

                else:
                    try:
                        with st.spinner("Analyzing your question..."):
                            user_emb = model.encode([prompt])
                            scores = cosine_similarity(user_emb, embeddings)[0]
                            best_idx = int(np.argmax(scores))
                            best_score = scores[best_idx]

                            if best_score > 0.7:
                                row = df.iloc[best_idx]
                                text = f"**Topic: {row['topic'].title()}**\n\n{row['answer']}"
                                confidence_msg = f"\n\n🧠 Confidence: {best_score:.2f}"
                                final_reply = text + confidence_msg
                            else:
                                search_query = rephrase_query(prompt)
                                encoded = urllib.parse.quote_plus(search_query)
                                google_url = f"https://www.google.com/search?q={encoded}"
                                final_reply = (
                                    "🤖 I couldn't find a good match.\n\n"
                                    f'🔎 <a href="{google_url}" target="_blank">Search on Google</a><br>'
                                    "📧 Contact safety team: safety@company.com"
                                )

                            with st.chat_message("assistant"):
                                st.markdown(final_reply, unsafe_allow_html=True)

                            st.session_state.messages.append({"role": "assistant", "content": final_reply})

                    except Exception as e:
                        error_msg = "⚠️ We're having technical issues. Please email safety@company.com."
                        with st.chat_message("assistant"):
                            st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
else:
    st.info("📁 Upload an Excel file to start chatting.")
