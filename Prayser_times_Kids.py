import streamlit as st
import datetime
import json
import os
import requests

# --- Page config for kid-friendly theme ---
st.set_page_config(
    page_title="Kids' Prayer Times",
    page_icon="🌙",
    layout="centered"
)

# --- Custom CSS for style ---
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; color: #FF6B6B; }
    .highlight { background-color: #FFF3CD; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Emoji/icons for prayers ---
prayer_emojis = {
    "Fajr": "🌙",
    "Dhuhr": "🌞",
    "Asr": "🌤️",
    "Maghrib": "🌆",
    "Isha": "🌃"
}

# --- Function to fetch prayer times using Aladhan API ---
def fetch_prayer_times(city="Cairo", country="Egypt"):
    url = "http://api.aladhan.com/v1/timingsByCity"
    params = {
        "city": city,
        "country": country,
        "method": 2  # ISNA
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        timings = data["data"]["timings"]
        return {
            "Fajr": timings["Fajr"],
            "Dhuhr": timings["Dhuhr"],
            "Asr": timings["Asr"],
            "Maghrib": timings["Maghrib"],
            "Isha": timings["Isha"]
        }
    except Exception as e:
        st.error("Failed to fetch prayer times.")
        return {
            "Fajr": "N/A",
            "Dhuhr": "N/A",
            "Asr": "N/A",
            "Maghrib": "N/A",
            "Isha": "N/A"
        }

# --- User input for location ---
st.sidebar.title("📍 Location Settings")
city = st.sidebar.text_input("City", "Cairo")
country = st.sidebar.text_input("Country", "Egypt")
if st.sidebar.button("Update Prayer Times"):
    st.session_state['prayer_times'] = fetch_prayer_times(city, country)

# --- Load or set default prayer times ---
if 'prayer_times' not in st.session_state:
    st.session_state['prayer_times'] = fetch_prayer_times(city, country)

prayer_times = st.session_state['prayer_times']

# --- Initialize session state for checkboxes ---
if 'checked' not in st.session_state:
    st.session_state.checked = {}

# --- Initialize checkboxes session state for each prayer ---
for prayer in prayer_times:
    if prayer not in st.session_state.checked:
        st.session_state.checked[prayer] = False

# --- Load prayer history ---
HISTORY_FILE = "prayer_history.json"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump({"completed_dates": []}, f)

with open(HISTORY_FILE, "r") as f:
    history_data = json.load(f)
    completed_dates = history_data["completed_dates"]

# --- Today's date ---
today = datetime.date.today()
today_str = str(today)

# --- Title and header ---
st.title("🌙 Kids' Prayer Times Tracker")
st.markdown("---")

# --- Display prayer times with checkboxes ---
for prayer, time in prayer_times.items():
    col1, col2, col3 = st.columns([1, 3, 3])
    with col1:
        st.markdown(f"<span class='big-font'>{prayer_emojis.get(prayer, '🕋')}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**{prayer}**\n\n`{time}`")
    with col3:
        st.session_state.checked[prayer] = st.checkbox(
            f"Done with {prayer}!",
            value=st.session_state.checked.get(prayer, False),
            key=prayer
        )

# --- Progress Tracker ---
completed = sum(st.session_state.checked.values())
total = len(prayer_times)
progress = completed / total

# --- Update history if day is complete ---
if completed == total and today_str not in completed_dates:
    completed_dates.append(today_str)
    with open(HISTORY_FILE, "w") as f:
        json.dump({"completed_dates": completed_dates}, f)

# --- Show progress ---
st.markdown("---")
st.subheader("Today's Progress 🎉")
st.markdown(f"**{completed}/{total} Prayers Completed**")
st.progress(progress)

if completed == total:
    st.balloons()
    st.success("🌟 Amazing! You finished all prayers today!")
    st.write("👏" * 10 + " YAY! " + "👏" * 10)

# --- Days Completed Summary ---
st.markdown("---")
st.subheader("Total Days Completed")
st.markdown(f"""
    <div class="highlight">
        🏆 **{len(completed_dates)} day{'s' if len(completed_dates) != 1 else ''}** where you prayed all 5 times!
    </div>
""", unsafe_allow_html=True)

# --- Timeline Display ---
st.markdown("---")
st.subheader("Day Timeline")
timeline = "🕒 " + " — ".join([f"{prayer_emojis.get(prayer, '')} {time}" for prayer, time in prayer_times.items()])
st.markdown(f"`{timeline}`")

# --- Reset Button ---
if st.button("Reset Checkboxes"):
    for prayer in prayer_times:
        st.session_state.checked[prayer] = False
    st.rerun()
