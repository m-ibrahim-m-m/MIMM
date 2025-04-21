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
def fetch_prayer_times(city="New York", country="USA"):
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
city = st.sidebar.text_input("City", "New York")
country = st.sidebar.text_input("Country", "USA")
if st.sidebar.button("Update Prayer Times"):
    st.session_state['prayer_times'] = fetch_prayer_times(city
