import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import urllib.parse

@st.cache_resource
def load_chatbot():
    return SafetyInstructionChatbot()

class SafetyInstructionChatbot:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.topic_phrases = {
            "fire safety": ["fire safety", "fire drill", "how to evacuate", "fire alarm protocol",
                            "what to do in case of fire", "fire extinguisher use"],
            "first aid": ["first aid", "CPR help", "how to stop bleeding", "emergency response",
                          "how to use AED", "basic life support"],
            "chemical spill": ["chemical spill", "hazardous material", "chemical accident",
                               "spill kit instructions", "toxic leak response"],
            "ppe usage": ["ppe usage", "how to wear protective equipment", "safety gear",
                          "mask gloves goggles", "using personal protective equipment"],
            "ppe training": ["ppe training", "ppe education", "how to train on safety gear", "personal protective equipment training",
                            "ppe instructions", "كيفية التدريب على معدات الوقاية الشخصية"],
            "earthquake drill": ["earthquake drill", "earthquake safety", "what to do during earthquake",
                                 "drop cover hold", "earthquake response steps"]
        }
        self.responses = {
            "fire safety": {
                "text": "🔥 Fire Safety Protocol:\n1. Activate nearest fire alarm\n2. Evacuate using marked exits\n3. Use fire extinguisher only if safe\n4. Gather at designated assembly point\n\nℹ️ For more information, please check our safety portal.",
                "video": ""
                ,}
                "first aid": {
                "text": "🩹 First Aid Basics:\n1. Check scene safety\n2. Call emergency services\n3. Perform CPR if trained\n4. Use AED if available\n5. Stop bleeding with clean cloth\n\nℹ️ For more information, please check our safety portal.",
                "video": ""
            },
            "chemical spill": {
                "text": "⚠️ Chemical Spill Response:\n1. Evacuate immediate area\n2. Alert trained personnel\n3. Avoid contact with substance\n4. Use spill kit if trained\n5. Report incident to supervisor\n\nℹ️ For more information, please check our safety portal.",
                "video": ""
            },
            "ppe usage": {
                "text": "🛡️ تشمل هذه المواصفات ما يلي:\n\n- القيود على نوعية الملابس في اماكن العمل\n- حماية الرأس\n- حماية العين والوجة\n- حماية الجسم\n- الوقاية من السقوط\n- حماية السمع\n- حماية اليد والذراع \n- معدات حماية الجهاز التنفسي \n - ملابس عالية الوضوح\n\n- استخدام معدات الوقاية الشخصية\n يجب اتخاذ جميع الخطوات المنطقية لضمان استخدام أي معدات الوقاية الشخصية المقدمة للموظفين بشكل صحيح. يجب توثيق إجراءات الاستخدام وإتاحتها لمن يحتاجون إليها.يجب على كل موظف استخدام أي معدات وقاية شخصية مقدمة له وفقًا لتعليمات وتدريب تم تلقيه بشأن استخدام معدات الوقاية الشخصية.\n\nℹ️ لمزيد من المعلومات، يرجى زيارة بوابة السلامة الخاصة بنا.",
                "video": "https://www.youtube.com/watch?v=xkMdLcB_vpE"
            },
            "ppe training": {
                "text": "🎓 **التدريب والتعليم على معدات الوقاية الشخصية (PPE)**\n\n"
                        "**أولاً: التدريب النظري**\n"
                        "- شرح المخاطر:\n"
                        "  - تحديد المخاطر في بيئة العمل\n"
                        "  - سبب الحاجة إلى معدات الوقاية\n"
                        "- تشغيل وأداء المعدات:\n"
                        "  - كيفية تشغيلها وأداؤها وحدود فعاليتها\n"
                        "- الاختيار والاستخدام والتخزين:\n"
                        "  - تعليمات اختيار المعدات المناسبة واستخدامها وتخزينها\n"
                        "- إجراءات التشغيل المكتوبة:\n"
                        "  - مثل تصاريح العمل المتعلقة بـ PPE\n"
                        "- العوامل المؤثرة على الفعالية:\n"
                        "  - مثل الحالة الصحية وظروف العمل والتركيب غير الصحيح\n"
                        "- التعرف على العيوب والإبلاغ عنها:\n"
                        "  - فحص المعدات وآلية التبليغ عن الأعطال\n"
                        "- السياسة التأديبية:\n"
                        "  - الإجراءات المتبعة عند عدم الالتزام بالاستخدام\n\n"
                        "**ثانياً: التدريب العملي**\n"
                        "- ارتداء وخلع المعدات بشكل صحيح\n"
                        "- الفحص والاختبار قبل الاستخدام\n"
                        "- الصيانة البسيطة بواسطة المستخدم\n"
                        "- التخزين الآمن للمعدات\n\n"
                        "**ثالثاً: التدريب الدوري**\n"
                        "- تنظيم تدريبات تنشيطية دورية\n"
                        "- الاحتفاظ بسجلات التدريب لكل فرد\n\nℹ️ لمزيد من المعلومات، يرجى زيارة بوابة السلامة الخاصة بنا.",
                "video": ""
            },
            "earthquake drill": {
                "text": "🌍 Earthquake Safety:\n1. Drop to hands and knees\n2. Cover head and neck\n3. Hold on to sturdy furniture\n4. Stay away from windows\n5. Evacuate when shaking stops\n\nℹ️ For more information, please check our safety portal.",
                "video": ""
            }
        }

        # Encode all topic phrases
        self.phrase_to_topic = {}
        all_phrases = []
        for topic, phrases in self.topic_phrases.items():
            for phrase in phrases:
                self.phrase_to_topic[phrase] = topic
                all_phrases.append(phrase)
        self.encoded_phrases = self.model.encode(all_phrases)
        self.all_phrases = all_phrases

    def rephrase_query(self, query):
        safety_keywords = [
            "safety procedures", "emergency guidelines", "workplace safety",
            "official protocol", "safety best practices", "OSHA standards"
        ]
        return f"{query} {np.random.choice(safety_keywords)}".strip()

    def get_response(self, user_input, threshold=0.75):
        try:
            user_embedding = self.model.encode(user_input.lower())
            similarities = cosine_similarity([user_embedding], self.encoded_phrases)[0]
            best_index = np.argmax(similarities)
            best_similarity = similarities[best_index]
            best_phrase = self.all_phrases[best_index]
            topic = self.phrase_to_topic[best_phrase]

            if best_similarity > threshold:
                response = self.responses[topic]
                confidence_msg = f"🧠 Matched topic: **{topic.title()}** (confidence: {best_similarity:.2f})"
                video_msg = f"\n📺 Watch demonstration: {response['video']}" if response["video"] else ""
                return response['text'], response["video"], confidence_msg, video_msg
            else:
                close_matches = difflib.get_close_matches(user_input.lower(), self.topic_phrases.keys(), cutoff=0.4)
                if close_matches:
                    suggestions = "\n".join(f"- {match.title()}" for match in close_matches)
                    return (
                        f"🔍 No exact match, but you might be referring to:\n{suggestions}",
                        None,
                        "❓ Need more help? Email: safety-support@company.com",
                        None
                    )
                else:
                    search_query = self.rephrase_query(user_input)
                    encoded_query = urllib.parse.quote_plus(search_query)
                    google_url = f"https://www.google.com/search?q={encoded_query}"
                    search_msg = f'🔎 I couldn’t find a good match. <a href="{google_url}" target="_blank">Click here to search Google</a><br>📧 Contact safety team: safety@company.com'
                    return (
                        "I couldn't find a response for that query.",
                        None,
                        search_msg,
                        None
                    )
        except Exception as e:
            return (
                "⚠️ We're having technical issues. Please email safety@company.com.", 
                None,
                None,
                None
            )


# Streamlit UI Setup
st.set_page_config(page_title="Safety Instruction Chatbot", page_icon="🛡️", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "🛡️ Welcome! How can I help with safety procedures today? Type 'menu' for options."}]
    dark_mode = st.checkbox("🌙 Enable Dark Mode", value=False)
    st.markdown("---")
    st.subheader("Supported Safety Topics")
    for topic in ["fire safety", "first aid", "chemical spill", "ppe usage","ppe training", "earthquake drill"]:
        st.markdown(f"- {topic.title()}")

if dark_mode:
    st.markdown("""
        <style>
            .stApp { background-color: #1e1e1e; color: white; }
            .stChatMessage { background-color: #2e2e2e; }
            a { color: #4dabf7; }
        </style>
    """, unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "🛡️ Welcome! I can help with:\n- Fire safety\n - PPE Training\n- First aid\n- PPE usage\n- Emergency drills\n\nType 'menu' for help."}]
    st.title("HSE Procedures Chatbot")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Handle user input
if prompt := st.chat_input("Type your safety question..."):
    chatbot = load_chatbot()
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

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
        with st.spinner("Analyzing your question..."):
            text, video_url, confidence_msg, video_msg = chatbot.get_response(prompt)
            
            # Build final response
            final_reply = text
            if confidence_msg:
                final_reply += f"\n\n{confidence_msg}"
            if video_msg:
                final_reply += f"\n{video_msg}"
            
            # Display response
            with st.chat_message("assistant"):
                st.markdown(final_reply, unsafe_allow_html=True)
                if video_url:
                    st.video(video_url)
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": final_reply})
