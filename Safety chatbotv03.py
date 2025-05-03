import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import urllib.parse
from langdetect import detect, LangDetectException
import fasttext  # For more accurate language detection

# Download fasttext model first: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin


@st.cache_resource
def load_chatbot():
    return SafetyInstructionChatbot()

class SafetyInstructionChatbot:
    def __init__(self):
        # Multilingual model for better Arabic support
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.lang_detector = fasttext.load_model('lid.176.bin')
        
        # Phrases organized by language
        self.topic_phrases = {
            "fire safety": {
                "en": ["fire safety", "fire drill", "evacuation procedure", "fire extinguisher use"],
                "ar": ["سلامة الحريق", "إجراءات الإخلاء", "استخدام طفاية الحريق"]
            },
            "ppe training": {
                "en": ["ppe training", "safety gear education", "protective equipment instruction"],
                "ar": ["تدريب على معدات الوقاية الشخصية", "تعليمات استخدام معدات السلامة"]
            },
            "ppe usage": {
                "en": ["ppe usage", "how to wear safety gear", "proper equipment use"],
                "ar": ["استخدام معدات الوقاية الشخصية", "طريقة ارتداء معدات الحماية"]
            }
        }

        # Responses organized by language
        self.responses = {
            "fire safety": {
                "en": {
                    "text": "🔥 Fire Safety Protocol:\n1. Activate nearest fire alarm...",
                    "video": "https://www.youtube.com/watch?v=GVBamXXVD30"
                },
                "ar": {
                    "text": "🔥 إجراءات سلامة الحريق:\n1. تفعيل إنذار الحريق الأقرب...",
                    "video": "https://www.youtube.com/watch?v=arabic_fire_video"
                }
            },
            "ppe training": {
                "en": {
                    "text": "🛡️ PPE Training:\n1. Theoretical safety principles...",
                    "video": ""
                },
                "ar": {
                    "text": "🛡️ تدريب معدات الوقاية الشخصية:\n1. المبادئ النظرية للسلامة...",
                    "video": "https://www.youtube.com/watch?v=xkMdLcB_vpE"
                }
            }
        }

        # Create unified phrase list with language metadata
        self.all_phrases = []
        self.phrase_metadata = []
        for topic, lang_phrases in self.topic_phrases.items():
            for lang, phrases in lang_phrases.items():
                for phrase in phrases:
                    self.all_phrases.append(phrase)
                    self.phrase_metadata.append({
                        "topic": topic,
                        "language": lang,
                        "original_phrase": phrase
                    })
        
        self.encoded_phrases = self.model.encode(self.all_phrases)

    def detect_language(self, text):
        try:
            predictions = self.lang_detector.predict(text.replace('\n', ' '), k=1)
            return predictions[0][0].replace('__label__', '')
        except:
            return 'en'

    def get_response(self, user_input, threshold=0.72):
        try:
            # Detect input language
            input_lang = self.detect_language(user_input)
            
            # Encode query
            user_embedding = self.model.encode(user_input)
            
            # Calculate similarities
            similarities = cosine_similarity([user_embedding], self.encoded_phrases)[0]
            best_index = np.argmax(similarities)
            best_similarity = similarities[best_index]
            best_phrase_meta = self.phrase_metadata[best_index]
            
            if best_similarity > threshold:
                # Get response in detected language
                response = self.responses[best_phrase_meta["topic"]].get(input_lang,
                    self.responses[best_phrase_meta["topic"]]["en"]  # Fallback to English
                )
                
                confidence_msg = {
                    "en": f"🧠 Matched: {best_phrase_meta['topic'].title()} (Confidence: {best_similarity:.2f})",
                    "ar": f"🧠 أفضل تطابق: {best_phrase_meta['topic'].title()} (الثقة: {best_similarity:.2f})"
                }.get(input_lang, "")
                
                return {
                    "text": response["text"],
                    "video": response["video"],
                    "language": input_lang,
                    "confidence": confidence_msg
                }
            else:
                # Generate suggestions in detected language
                lang_phrases = [
                    p["original_phrase"] for p in self.phrase_metadata 
                    if p["language"] == input_lang
                ]
                close_matches = difflib.get_close_matches(
                    user_input.lower(), 
                    lang_phrases,
                    n=3,
                    cutoff=0.4
                )
                
                if close_matches:
                    suggestions = "\n".join(f"- {match}" for match in close_matches)
                    return {
                        "text": {
                            "en": f"🔍 Did you mean:\n{suggestions}",
                            "ar": f"🔍 هل تقصد:\n{suggestions}"
                        }.get(input_lang, ""),
                        "language": input_lang
                    }
                else:
                    search_query = urllib.parse.quote_plus(user_input)
                    return {
                        "text": {
                            "en": f'🔎 No matches found. <a href="https://www.google.com/search?q={search_query}" target="_blank">Search Google</a>',
                            "ar": f'🔎 لم يتم العثور على تطابق. <a href="https://www.google.com/search?q={search_query}" target="_blank">ابحث في جوجل</a>'
                        }.get(input_lang, ""),
                        "language": input_lang
                    }
                    
        except Exception as e:
            return {
                "text": {
                    "en": "⚠️ Technical error. Please contact safety@company.com",
                    "ar": "⚠️ خطأ فني. يرجى الاتصال بـ safety@company.com"
                },
                "language": "en"
            }

# Streamlit UI
st.set_page_config(page_title="Multilingual Safety Chatbot", page_icon="🌍", layout="wide")

# RTL CSS
st.markdown("""
    <style>
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    .stChatMessage.rtl .message-content {
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "🌍 Welcome! Ask about:\n- Fire safety\n- PPE training\n- Safety equipment usage\n\nمرحبا! اسأل عن:\n- سلامة الحريق\n- تدريب المعدات\n- استخدام معدات الوقاية",
        "lang": "bilingual"
    }]

# Sidebar
with st.sidebar:
    st.header("الإعدادات / Settings")
    if st.button("🗑️ مسح المحادثة / Clear Chat"):
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "🌍 Welcome! Ask about:\n- Fire safety\n- PPE training\n- Safety equipment usage\n\nمرحبا! اسأل عن:\n- سلامة الحريق\n- تدريب المعدات\n- استخدام معدات الوقاية",
            "lang": "bilingual"
        }]
    st.download_button("📥 حفظ المحادثة / Save Chat", "\n".join(m["content"] for m in st.session_state.messages))

# Chat display
for msg in st.session_state.messages:
    css_class = "rtl" if msg.get("lang") == "ar" else ""
    with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else None):
        st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# Input handling
if prompt := st.chat_input("اكتب سؤالاً هنا / Type your question..."):
    chatbot = load_chatbot()
    st.session_state.messages.append({"role": "user", "content": prompt, "lang": "user"})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("جاري البحث... / Searching..."):
        response = chatbot.get_response(prompt)
        
        # Prepare response
        response_text = response["text"]
        if "confidence" in response:
            response_text += f"\n\n{response['confidence']}"
        
        # Apply RTL if Arabic
        css_class = "rtl" if response.get("language") == "ar" else ""
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(f'<div class="{css_class}">{response_text}</div>', unsafe_allow_html=True)
            if response.get("video"):
                st.video(response["video"])
        
        # Store in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "lang": response.get("language", "en")
        })
